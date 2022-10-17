import os
from tool.utils import available_devices,format_devices
#set device
device = available_devices(threshold=10000,n_devices=4)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
from tool.reproducibility import set_seed
from tool.utils import dict2namespace
import yaml
import torch
from runners.egsde import EGSDE
from tool.interact import set_logger
from profiles.multi_afhq.args import argsall

def run_egsde(args):
    #config
    with open(args.config_path, "r") as f:
        config_ = yaml.safe_load(f)
    config = dict2namespace(config_)
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    runner = EGSDE(args, config)
    runner.egsde()

if __name__ == "__main__":
    # multi-domain translation (any animal to dog) in Appendix D

    # args
    args = argsall
    set_seed(args.seed)

    #cat2dog
    task = 'cat2dog'
    args.testdata_path = 'data/afhq/val/cat'
    args.samplepath = os.path.join('runs', 'mutli_afhq',task)
    os.makedirs(args.samplepath, exist_ok=True)
    set_logger(args.samplepath, 'sample.txt')
    run_egsde(args)

    # wild2dog
    task = 'wild2dog'
    args.testdata_path = 'data/afhq/val/wild'
    args.samplepath = os.path.join('runs','mutli_afhq',task)
    os.makedirs(args.samplepath, exist_ok=True)
    set_logger(args.samplepath, 'sample.txt')
    run_egsde(args)













