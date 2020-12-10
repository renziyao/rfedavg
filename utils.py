import torch

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