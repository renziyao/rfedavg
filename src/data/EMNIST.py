import torchvision.datasets
import torchvision.transforms as transforms
from src.data.utils import *

root_dir = './data/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.EMNIST(
    root_dir, 
    train=True, 
    transform=transform, 
    split='byclass',
    download=True,
)
test_dataset = torchvision.datasets.EMNIST(
    root_dir, 
    train=False, 
    transform=transform, 
    split='byclass',
    download=True,
)

def non_iid_shard(params):
    num_user = params['Trainer']['n_clients']
    shard_per_user = params['Dataset']['shard_per_user']
    dataset_split = split_dataset_by_shard(
        train_dataset, 
        test_dataset, 
        num_user,
        shard_per_user,
    )
    return dataset_split, test_dataset

def non_iid_percent(params):
    num_user = params['Trainer']['n_clients']
    s = params['Dataset']['s']
    dataset_split = split_dataset_by_percent(train_dataset, test_dataset, s, num_user)
    return dataset_split, test_dataset