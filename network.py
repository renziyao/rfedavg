import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.alexnet(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        model = models.alexnet(pretrained=False)
        self.classifier = nn.Sequential(
            model.classifier[0:3],
            model.classifier[3:6],
            model.classifier[6:],
        )
        self.fc = nn.Linear(1000, 31)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f1 = self.classifier[0](x)
        f2 = self.classifier[1](f1)
        f3 = self.classifier[2](f2)
        prediction = self.fc(f3)
        return prediction, [f1, f2, f3]