import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred = self.fc3(x)
        return pred, x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f = self.classifier(x)
        prediction = self.fc(f)
        return prediction, f
