import importlib

import yaml
from src.utils import read_options, print_params


if __name__ == '__main__':
    params = read_options()
    print_params(params)
    server = importlib.import_module(
        'src.trainers.%s' % params['Trainer']['name']
    ).Server(params)
    server.train()
