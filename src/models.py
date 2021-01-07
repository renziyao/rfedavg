import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def parameters_to_tensor(self):
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        params = torch.cat(params, 0)
        return params
    
    def tensor_to_parameters(self, tensor):
        p = 0
        for param in self.parameters():
            shape = param.shape
            delta = 1
            for item in shape: delta *= item
            param.data = tensor[p: p + delta].view(shape).detach().clone()
            p += delta

class LeNet(BaseModule):
    def __init__(self, input_shape=(1, 28, 28), cls_num=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, cls_num)

    def forward(self, x, features=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred = self.fc3(x)
        if features: return pred, x
        else: return pred

class FedAvgCNN(BaseModule):
    def __init__(self, input_shape=(1, 28, 28), cls_num=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, cls_num)

    def forward(self, x, features=False):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        pred = self.fc2(out)
        if features: return pred, out
        else: return pred
