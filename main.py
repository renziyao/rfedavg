import importlib

from src.utils import *


if __name__ == '__main__':
    params = read_options()
    if 'Output' in params: redirect_stdout(params['Output'])
    print_params(params)
    set_seed(params['Trainer']['seed'])
    server = importlib.import_module(
        'src.trainers.%s' % params['Trainer']['name']
    ).Server(params)
    server.train()
