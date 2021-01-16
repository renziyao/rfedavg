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

def nlp_collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    labels, texts = zip(*batch)
    labels = list(labels)
    texts = list(texts)
    max_length = len(texts[0])
    length = []
    for i, text in enumerate(texts):
        length.append(len(text))
        if len(text) == length: continue
        texts[i] = torch.cat([text, torch.zeros(max_length - len(text), dtype=torch.long)])
    length = torch.tensor(length)
    texts = torch.stack(texts)
    inputs = torch.cat([texts, length.view(-1, 1)], dim=1)
    return inputs, torch.stack(labels)