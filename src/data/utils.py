import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

def sort_idx(idx, dataset_target, labels_count):
    sorted_idx_lst = [
        idx[dataset_target[idx] == i] for i in range(labels_count)
    ]
    sorted_idx = np.concatenate(sorted_idx_lst, 0)
    return sorted_idx

def split_dataset_by_shard(train_dataset, test_dataset, num_user, shard_per_user):
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    trainset_data_targets = next(iter(train_loader))
    testset_data_targets = next(iter(test_loader))
    trainset_data = trainset_data_targets[0].numpy()
    trainset_targets = trainset_data_targets[1].numpy()
    testset_data = testset_data_targets[0].numpy()
    testset_targets = testset_data_targets[1].numpy()

    trainset_idx = np.array([i for i in range(len(train_dataset))])
    testset_idx = np.array([i for i in range(len(test_dataset))])
    labels_count = len(train_dataset.class_to_idx)

    trainset_idx = sort_idx(trainset_idx, trainset_targets, labels_count)
    testset_idx = sort_idx(testset_idx, testset_targets, labels_count)
    
    total_shard = num_user * shard_per_user
    shards = []

    p_train = 0
    p_test = 0
    delta_train = trainset_idx.shape[0] // total_shard
    delta_test = testset_idx.shape[0] // total_shard
    for _ in range(total_shard):
        train_X = torch.tensor(
            trainset_data[
                trainset_idx[
                    p_train: p_train + delta_train
                ]
            ]
        )
        train_Y = torch.tensor(
            trainset_targets[
                trainset_idx[
                    p_train: p_train + delta_train
                ]
            ],
        )
        test_X = torch.tensor(
            testset_data[
                testset_idx[
                    p_test: p_test + delta_test
                ]
            ],
        )
        test_Y = torch.tensor(
            testset_targets[
                testset_idx[
                    p_test: p_test + delta_test
                ]
            ],
        )
        shards.append([
            train_X,
            train_Y,
            test_X,
            test_Y,
        ])
        p_train += delta_train
        p_test += delta_test
    
    shards_idx = np.array([i for i in range(len(shards))])
    shards_idx = np.random.shuffle(shards_idx)

    p_shard = 0
    dataset_split = []
    for _ in range(num_user):
        user_shards = shards[p_shard: p_shard + shard_per_user]
        train_X = torch.tensor(
            np.concatenate(
                [item[0] for item in user_shards], 
                axis=0, 
            )
        )
        train_Y = torch.tensor(
            np.concatenate(
                [item[1] for item in user_shards], 
                axis=0, 
            )
        )
        test_X = torch.tensor(
            np.concatenate(
                [item[2] for item in user_shards], 
                axis=0, 
            )
        )
        test_Y = torch.tensor(
            np.concatenate(
                [item[3] for item in user_shards], 
                axis=0, 
            )
        )
        dataset_split.append(
            {
                'train': TensorDataset(train_X, train_Y),
                'test': TensorDataset(test_X, test_Y),
            }
        )
        p_shard += shard_per_user
    return dataset_split

def split_dataset_by_percent(train_dataset, test_dataset, s: float, num_user: int):
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    trainset_data_targets = next(iter(train_loader))
    testset_data_targets = next(iter(test_loader))
    trainset_data = trainset_data_targets[0].numpy()
    trainset_targets = trainset_data_targets[1].numpy()
    testset_data = testset_data_targets[0].numpy()
    testset_targets = testset_data_targets[1].numpy()

    len_train_iid = round(s * len(train_dataset))
    len_test_iid = round(s * len(test_dataset))
    trainset_iid_idx = np.array([i for i in range(len_train_iid)])
    trainset_niid_idx = np.array([
        i for i in range(len_train_iid, len(train_dataset))
    ])
    testset_iid_idx = np.array([i for i in range(len_test_iid)])
    testset_niid_idx = np.array([
        i for i in range(len_test_iid, len(test_dataset))
    ])
    labels_count = len(train_dataset.class_to_idx)
    
    if len(trainset_niid_idx) > 0:
        trainset_niid_idx = sort_idx(trainset_niid_idx, trainset_targets, labels_count)
    if len(testset_niid_idx) > 0:
        testset_niid_idx = sort_idx(testset_niid_idx, testset_targets, labels_count)

    p_train_iid = 0
    p_train_niid = 0
    p_test_iid = 0
    p_test_niid = 0
    delta_train_iid = trainset_iid_idx.shape[0] // num_user
    delta_train_niid = trainset_niid_idx.shape[0] // num_user
    delta_test_iid = testset_iid_idx.shape[0] // num_user
    delta_test_niid = testset_niid_idx.shape[0] // num_user
    dataset_split = []
    for userid in range(num_user):
        train_X_lst = []
        train_Y_lst = []
        test_X_lst = []
        test_Y_lst = []
        if delta_train_iid > 0:
            train_X_lst.append(
                trainset_data[
                    trainset_iid_idx[
                        p_train_iid: p_train_iid + delta_train_iid
                    ]
                ]
            )
            train_Y_lst.append(
                trainset_targets[
                    trainset_iid_idx[
                        p_train_iid: p_train_iid + delta_train_iid
                    ]
                ]
            )
        if delta_train_niid > 0:
            train_X_lst.append(
                trainset_data[
                    trainset_niid_idx[
                        p_train_niid: p_train_niid + delta_train_niid
                    ]
                ]
            )
            train_Y_lst.append(
                trainset_targets[
                    trainset_niid_idx[
                        p_train_niid: p_train_niid + delta_train_niid
                    ]
                ]
            )
        if delta_test_iid > 0:
            test_X_lst.append(
                testset_data[
                    testset_iid_idx[
                        p_test_iid: p_test_iid + delta_test_iid
                    ]
                ]
            )
            test_Y_lst.append(
                testset_targets[
                    testset_iid_idx[
                        p_test_iid: p_test_iid + delta_test_iid
                    ]
                ]
            )
        if delta_test_niid > 0:
            test_X_lst.append(
                testset_data[
                    testset_niid_idx[
                        p_test_niid: p_test_niid + delta_test_niid
                    ]
                ]
            )
            test_Y_lst.append(
                testset_targets[
                    testset_niid_idx[
                        p_test_niid: p_test_niid + delta_test_niid
                    ]
                ]
            )
            
        train_X = torch.tensor(
            np.concatenate(train_X_lst, axis=0)
        )
        train_Y = torch.tensor(
            np.concatenate(train_Y_lst, axis=0)
        )
        test_X = torch.tensor(
            np.concatenate(test_X_lst, axis=0)
        )
        test_Y = torch.tensor(
            np.concatenate(test_Y_lst, axis=0)
        )
        dataset_split.append(
            {
                'train': TensorDataset(train_X, train_Y),
                'test': TensorDataset(test_X, test_Y),
            }
        )
        p_train_iid += delta_train_iid
        p_train_niid += delta_train_niid
        p_test_iid += delta_test_iid
        p_test_niid += delta_test_niid
    return dataset_split