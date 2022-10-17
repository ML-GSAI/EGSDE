import os
import logging
import numpy as np
import torch
import torch.utils.data as data
from models.ddpm import Model
from datasets import get_dataset,rescale,inverse_rescale
import torchvision.utils as tvu
from functions.denoising import egsde_sample
from guided_diffusion.script_util import create_model,create_dse
from functions.resizer import Resizer
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def truncation(global_step,ratio=0.5):
    part = int(global_step * ratio)
    weight_l = torch.zeros(part).reshape(-1, 1)
    weight_r = torch.ones(global_step - part).reshape(-1, 1)
    weight = torch.cat((weight_l, weight_r), dim=0)
    return weight

class EGSDE(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device


        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def egsde(self,):
        args, config = self.args, self.config
        #load SDE
        if args.diffusionmodel == 'ADM':
            model = create_model(image_size=config.data.image_size,
                                 num_class=config.model.num_class,
                                 num_channels=config.model.num_channels,
                                 num_res_blocks=config.model.num_res_blocks,
                                 learn_sigma=config.model.learn_sigma,
                                 class_cond=config.model.class_cond,
                                 attention_resolutions=config.model.attention_resolutions,
                                 num_heads=config.model.num_heads,
                                 num_head_channels=config.model.num_head_channels,
                                 num_heads_upsample=config.model.num_heads_upsample,
                                 use_scale_shift_norm=config.model.use_scale_shift_norm,
                                 dropout=config.model.dropout,
                                 resblock_updown=config.model.resblock_updown,
                                 use_fp16=config.model.use_fp16,
                                 use_new_attention_order=config.model.use_new_attention_order)
            states = torch.load(args.ckpt)
            model.load_state_dict(states)
            model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.eval()
        elif args.diffusionmodel == 'DDPM':
            model = Model(config)
            states = torch.load(self.args.ckpt)
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states, strict=True)
            model.eval()
        else:
            raise ValueError(f"unsupported diffusion model")

        #load domain-specific feature extractor
        dse = create_dse(image_size=config.data.image_size,
                         num_class=config.dse.num_class,
                         classifier_use_fp16=config.dse.classifier_use_fp16,
                         classifier_width=config.dse.classifier_width,
                         classifier_depth=config.dse.classifier_depth,
                         classifier_attention_resolutions=config.dse.classifier_attention_resolutions,
                         classifier_use_scale_shift_norm=config.dse.classifier_use_scale_shift_norm,
                         classifier_resblock_updown=config.dse.classifier_resblock_updown,
                         classifier_pool=config.dse.classifier_pool,
                         phase=args.phase)
        states = torch.load(args.dsepath)
        dse.load_state_dict(states)
        dse.to(self.device)
        dse = torch.nn.DataParallel(dse)
        dse.eval()

        #load domain-independent feature extractor
        shape = (args.batch_size, 3, config.data.image_size, config.data.image_size)
        shape_d = (
            args.batch_size, 3, int(config.data.image_size / args.down_N), int(config.data.image_size / args.down_N))
        down = Resizer(shape, 1 / args.down_N).to(self.device)
        up = Resizer(shape_d, args.down_N).to(self.device)
        die = (down, up)

        #create dataset
        dataset = get_dataset(phase=args.phase,image_size= config.data.image_size, data_path = args.testdata_path)
        data_loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )

        for i, (y, name) in enumerate(data_loader):
            logging.info(f'batch:{i}/{len(dataset) / args.batch_size}')
            n = y.size(0)
            y0 = rescale(y).to(self.device)
            #let x0 be source image
            x0 = y0
            #args.sample_step: the times for repeating EGSDE(usually set 1) (see Appendix A.2)
            for it in range(args.sample_step):
                e = torch.randn_like(y0)
                total_noise_levels = args.t
                a = (1 - self.betas).cumprod(dim=0)
                # the start point M: y ∼ qM|0(y|x0)
                y = y0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                for i in reversed(range(total_noise_levels)):
                    t = (torch.ones(n) * i).to(self.device)
                    #sample perturbed source image from the perturbation kernel: x ∼ qs|0(x|x0)
                    xt = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                    # egsde update (see VP-EGSDE in Appendix A.3)
                    y_ = egsde_sample(y=y, dse=dse,ls=args.ls,die=die,li=args.li,t=t,model=model,
                                        logvar=self.logvar,betas=self.betas,xt=xt,s1=args.s1,s2=args.s2, model_name = args.diffusionmodel)
                    y = y_
                y0 = y
                y = inverse_rescale(y)
                #save image
                for b in range(n):
                    os.makedirs(os.path.join(self.args.samplepath, str(it)), exist_ok=True)
                    tvu.save_image(
                        y[b], os.path.join(self.args.samplepath, str(it), name[b])
                    )
        logging.info('Finshed sampling.')
