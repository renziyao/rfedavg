import torch
import torch.utils
from torch.utils.data.dataset import TensorDataset
import torchvision.datasets
import torchvision.transforms as transforms
import dalib.vision.datasets
import torch.utils.data
import random
import numpy as np

def get_mnist_train_niid(root_dir, NUM_USER=50):
    if NUM_USER % 5 != 0: raise Exception("NUM_USER must be 5 * x")
    SPLIT_DIGIT = NUM_USER * 2 // 10

    trainset = torchvision.datasets.MNIST(
        root=root_dir, 
        train=True, 
        download=True,
    )
    trainset_data = trainset.data
    trainset_targets = trainset.targets

    trainset_data = trainset_data.unsqueeze(1)
    trainset_data = trainset_data.numpy() / 255
    
    trainset_digits = []
    for i in range(10):
        idx = trainset_targets == i
        trainset_digits.append(trainset_data[idx])
    
    trainset_digits_split = []
    for digit in trainset_digits:
        length, remain = len(digit) // SPLIT_DIGIT, len(digit) % SPLIT_DIGIT
        p = 0
        digit_split = []
        while p < len(digit):
            digit_split.append(digit[p: p + length + (remain > 0)])
            p += length + (remain > 0)
            remain = max(0, remain - 1)
        trainset_digits_split.append(digit_split)
    
    train_X = [[] for _ in range(NUM_USER)]
    train_Y = [[] for _ in range(NUM_USER)]
    for userid in range(NUM_USER):
        available_digits = [
            i for i in range(10) if len(trainset_digits_split[i]) > 0
        ]
        if len(available_digits) >= 2:
            idx = random.sample(
                available_digits, 
                2, 
            )
        else:
            idx = [available_digits[0], available_digits[0]]
        for item in idx:
            l = len(trainset_digits_split[item][-1])
            train_X[userid].append(
                torch.tensor(
                    trainset_digits_split[item].pop(), 
                    dtype=torch.float32,
                )
            )
            train_Y[userid].append(
                torch.tensor(
                    item * np.ones(l),
                    dtype=torch.long,
                )
            )
    
    MNIST_split = []
    for userid in range(NUM_USER):
        MNIST_split.append(
            TensorDataset(
                torch.cat(train_X[userid], dim=0),
                torch.cat(train_Y[userid], dim=0)
            ),
        )
    
    testset = torchvision.datasets.MNIST(
        root=root_dir, 
        train=False, 
        download=True,
    )
    testset_data = testset.data
    testset_targets = testset.targets

    testset_data = testset_data.unsqueeze(1)
    testset_data = testset_data.numpy() / 255

    test_dataset = TensorDataset(
        torch.tensor(
            testset_data,
            dtype=torch.float32,
        ),
        testset_targets,
    )
    return MNIST_split, test_dataset


def split_dataset(dataset, n_clients):
    sample_per_client = len(dataset) // n_clients
    sample_remain = len(dataset) % n_clients
    client_trainsets = torch.utils.data.random_split(
        dataset,
        [sample_per_client + (i < sample_remain) for i in range(n_clients)]
    )
    return client_trainsets

def generate_dataset(task):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((256, 256), 0),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256), 0),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = dalib.vision.datasets.Office31(
        root='./data/office31/',
        task=task,
        transform=train_transform,
        download=True,
    )
    testset = dalib.vision.datasets.Office31(
        root='./data/office31/',
        task=task,
        transform=val_transform,
        download=True,
    )
    return trainset, testset