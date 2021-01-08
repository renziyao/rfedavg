import importlib

from src.utils import read_options, print_params, set_seed


if __name__ == '__main__':
    params = read_options()
    print_params(params)
    set_seed(params['Trainer']['seed'])
    server = importlib.import_module(
        'src.trainers.%s' % params['Trainer']['name']
    ).Server(params)
    server.train()
