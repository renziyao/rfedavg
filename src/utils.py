import torch
import torch.nn as nn


def test_acc(net, testset, gpu):
    testloader = torch.utils.data.DataLoader(testset, batch_size=16)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(gpu)
            labels = labels.to(gpu)
            outputs, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def send_params(center_model, client_model):
    center_model_params = center_model.state_dict()
    for (name, params) in client_model.named_parameters():
        params.data = center_model_params[name].clone().detach()
    return

def aggregate_params(center_model, client_models):
    n = len(client_models)
    for (name, params) in center_model.named_parameters():
        params.data = torch.zeros_like(params.data)
        for i, client_model in enumerate(client_models):
            params.data += (1 / n) * \
                client_model.state_dict()[name]
    center_model.zero_grad()
    return

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

