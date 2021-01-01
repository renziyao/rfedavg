import torch
import torch.nn as nn
import yaml

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader: torch.utils.data.DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

class LinearMMD(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X_avg = torch.sum(X, dim=0) / X.shape[0]
        Y_avg = torch.sum(Y, dim=0) / Y.shape[0]
        dis = torch.norm(X_avg - Y_avg) ** 2
        return dis

def read_options():
    with open('config.yml', 'r') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)
    if 'T' not in params['Trainer']:
        params['Trainer']['T'] = params['Trainer']['total_epoch'] // params['Trainer']['E']
    return params

def print_params(params):
    print(yaml.dump(params, Dumper=yaml.CDumper))