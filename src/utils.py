import yaml
import torch
import torch.cuda
import torch.backends.cudnn
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def read_options():
    with open('config.yml', 'r') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)
    if 'Round' not in params['Trainer']:
        params['Trainer']['Round'] = params['Trainer']['total_epoch'] // params['Trainer']['E']
    return params

def print_params(params):
    print(yaml.dump(params, Dumper=yaml.CDumper))