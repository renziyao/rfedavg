from torch.utils.data import DataLoader

def split_by_class(train_dataset, test_dataset, split_digit):
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    trainset_data = next(iter(train_loader))[0].numpy()
    trainset_targets = next(iter(train_loader))[1].numpy()
    testset_data = next(iter(test_loader))[0].numpy()
    testset_targets = next(iter(test_loader))[1].numpy()

    trainset_classes = []
    testset_classes = []
    labels_count = len(train_dataset.class_to_idx)
    for i in range(labels_count):
        idx = trainset_targets == i
        trainset_classes.append(trainset_data[idx])
    for i in range(labels_count):
        idx = testset_targets == i
        testset_classes.append(testset_data[idx])
    
    trainset_classes_split = []
    testset_classes_split = []
    for cls in trainset_classes:
        length, remain = len(cls) // split_digit, len(cls) % split_digit
        p = 0
        class_split = []
        while p < len(cls):
            class_split.append(cls[p: p + length + (remain > 0)])
            p += length + (remain > 0)
            remain = max(0, remain - 1)
        trainset_classes_split.append(class_split)
    for cls in testset_classes:
        length, remain = len(cls) // split_digit, len(cls) % split_digit
        p = 0
        class_split = []
        while p < len(cls):
            class_split.append(cls[p: p + length + (remain > 0)])
            p += length + (remain > 0)
            remain = max(0, remain - 1)
        testset_classes_split.append(class_split)
    return trainset_classes_split, testset_classes_split