import torch
import argparse
from src.federated_learning import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = {}
    params['device'] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        '--clients', 
        type=int, 
        default=10, 
        help='number of clients',
    )
    parser.add_argument(
        '--epoch', 
        type=int, 
        default=5, 
        help='number of local epoch',
    )
    parser.add_argument(
        '--c', 
        type=float, 
        default=1.0, 
        help='clients sample rate',
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=100, 
        help='batch size',
    )
    parser.add_argument(
        '--totalepoch', 
        type=int, 
        default=500, 
        help='total epoch',
    )
    parser.add_argument(
        '--testinterval', 
        type=int, 
        default=1, 
        help='test interval',
    )
    parser.add_argument(
        '--debug', 
        type=bool, 
        default=True, 
        help='print debug info',
    )
    parser.add_argument(
        '--tradeoff', 
        type=float, 
        default=0.25, 
        help='regularization tradeoff parameter(lambda)',
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001, 
        help='learning rate',
    )
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0.9, 
        help='momentum',
    )
    parser.add_argument(
        '--weightdecay', 
        type=float, 
        default=0.0005, 
        help='weight decay',
    )
    args = parser.parse_args()
    params['n_clients'] = args.clients
    params['E'] = args.epoch
    params['C'] = args.c
    params['batch_size'] = args.batch
    params['T'] = args.totalepoch
    params['T'] //= params['E']
    params['TestInterval'] = args.testinterval
    params['DEBUG'] = args.debug
    params['lambda'] = args.tradeoff
    params['optimizer'] = {}
    params['optimizer']['lr'] = args.lr
    params['optimizer']['momentum'] = args.momentum
    params['optimizer']['weight_decay'] = args.weightdecay
    print(params)
    main(params)
