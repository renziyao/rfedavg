# Regularized FedAvg

#### trainer

FedAvg, FedProx, rFedAvg1, rFedAvg2, SCAFFOLD, qFedAvg

#### dataset

MNIST: 60k, 1x28x28

CIFAR10:60k, 3x32x32, resized 3x28x28

EMNIST: 814k, 1x28x28

#### divide

non_iid_percent: SCAFFOLD中数据切分方式，s%的数据打乱后切分，1-s%的数据排序后切分

non_iid_shard: on fedavg non iid convergence切分方式，直接排序后切分为shard，每个client分配1个或2个shard

#### model

FedAvgCNN, LeNet, FedAvg2NN, LogisticRegression

参数：input_shape: [c, h, w], cls_num: 最后一层个数
