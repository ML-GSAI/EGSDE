import os
import torch
import torch.nn as nn
def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError
class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)

def available_devices(threshold=5000,n_devices=None):
    memory = list(os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader'))
    mem = [int(x.strip()) for x in memory]
    devices = []
    for i in range(len(mem)):
        if mem[i]>threshold:
            devices.append(i)
    device = devices if n_devices is None else devices[:n_devices]
    return device

def format_devices(devices):
    if isinstance(devices, list):
        return ','.join(map(str,devices))

import argparse
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

