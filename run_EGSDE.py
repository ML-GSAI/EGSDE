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

def run_egsde(task):
    if task == 'cat2dog':
        from profiles.cat2dog.args import argsall
    if task == 'wild2dog':
        from profiles.wild2dog.args import argsall
    if task == 'male2female':
        from profiles.male2female.args import argsall

    # args
    args = argsall
    set_seed(args.seed)
    args.samplepath = os.path.join('runs', task)
    os.makedirs(args.samplepath, exist_ok=True)
    set_logger(args.samplepath, 'sample.txt')

    #config
    with open(args.config_path, "r") as f:
        config_ = yaml.safe_load(f)
    config = dict2namespace(config_)
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    runner = EGSDE(args, config)
    runner.egsde()

if __name__ == "__main__":
    task = 'male2female'
    run_egsde(task)










