import numpy as np
import random
import torch.utils
from torch.utils.data.dataset import TensorDataset
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data
from src.data.utils import split_by_class

def non_iid_1(params):
    root_dir = './data/'
    NUM_USER = params['Trainer']['n_clients']
    if NUM_USER % 10 != 0: raise Exception("NUM_USER must be 10 * x")
    split_digit = NUM_USER // 10

    transform = transforms.Compose([
        transforms.RandomCrop((28, 28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    train_dataset = torchvision.datasets.CIFAR10(
        root_dir, 
        train=True, 
        transform=transform, 
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root_dir, 
        train=False, 
        transform=transform, 
        download=True
    )

    trainset_digits_split, testset_digits_split = split_by_class(
        train_dataset, 
        test_dataset, 
        split_digit,
    )
    
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for userid in range(NUM_USER):
        available_digits = [
            i for i in range(10) if len(trainset_digits_split[i]) > 0
        ]
        idx = random.choice(available_digits)
        l = len(trainset_digits_split[idx][-1])
        train_X.append(
            torch.tensor(trainset_digits_split[idx].pop(), dtype=torch.float32)
        )
        train_Y.append(
            torch.tensor(idx * np.ones(l), dtype=torch.long)
        )
        l = len(testset_digits_split[idx][-1])
        test_X.append(
            torch.tensor(testset_digits_split[idx].pop(), dtype=torch.float32)
        )
        test_Y.append(
            torch.tensor(idx * np.ones(l), dtype=torch.long)
        )
    
    MNIST_split = []
    for userid in range(NUM_USER):
        MNIST_split.append(
            {
                'train': TensorDataset(train_X[userid], train_Y[userid]),
                'test': TensorDataset(test_X[userid], test_Y[userid]),
            }
        )

    return MNIST_split, test_dataset

def non_iid_2(params):
    root_dir = './data/'
    NUM_USER = params['Trainer']['n_clients']
    if NUM_USER % 5 != 0: raise Exception("NUM_USER must be 5 * x")
    split_digit = NUM_USER * 2 // 10

    transform = transforms.Compose([
        transforms.RandomCrop((28, 28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root_dir, 
        train=True, 
        transform=transform, 
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root_dir, 
        train=False, 
        transform=transform, 
        download=True
    )

    trainset_digits_split, testset_digits_split = split_by_class(
        train_dataset, 
        test_dataset, 
        split_digit,
    )
    
    train_X = [[] for _ in range(NUM_USER)]
    train_Y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_Y = [[] for _ in range(NUM_USER)]
    for userid in range(NUM_USER):
        available_digits = [
            i for i in range(10) if len(trainset_digits_split[i]) > 0
        ]
        if len(available_digits) >= 2:
            idxs = random.sample(available_digits, 2)
        else:
            idxs = [available_digits[0] for _ in range(2)]
        
        for idx in idxs:
            l = len(trainset_digits_split[idx][-1])
            train_X[userid].append(
                torch.tensor(trainset_digits_split[idx].pop(), dtype=torch.float32)
            )
            train_Y[userid].append(
                torch.tensor(idx * np.ones(l), dtype=torch.long)
            )
            l = len(testset_digits_split[idx][-1])
            test_X[userid].append(
                torch.tensor(testset_digits_split[idx].pop(), dtype=torch.float32)
            )
            test_Y[userid].append(
                torch.tensor(idx * np.ones(l), dtype=torch.long)
            )
    
    MNIST_split = []
    for userid in range(NUM_USER):
        MNIST_split.append(
            {
                'train': TensorDataset(
                            torch.cat(train_X[userid], dim=0), 
                            torch.cat(train_Y[userid], dim=0)
                         ),
                'test': TensorDataset(
                            torch.cat(test_X[userid], dim=0), 
                            torch.cat(test_Y[userid], dim=0)
                        ),
            }
        )

    return MNIST_split, test_dataset