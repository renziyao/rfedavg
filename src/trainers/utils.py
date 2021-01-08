import torch
import torch.nn as nn

class LinearMMD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X_avg = torch.sum(X, dim=0) / X.shape[0]
        Y_avg = torch.sum(Y, dim=0) / Y.shape[0]
        dis = torch.norm(X_avg - Y_avg) ** 2
        return dis