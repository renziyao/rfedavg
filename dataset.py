import torchvision.transforms as transforms
import dalib.vision.datasets


def generate_dataset():
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
    trainset_source = dalib.vision.datasets.Office31(
        root='./data/',
        task='A',
        transform=train_transform,
    )
    trainset_target = dalib.vision.datasets.Office31(
        root='./data/',
        task='W',
        transform=train_transform,
    )
    testset_source = dalib.vision.datasets.Office31(
        root='./data/',
        task='A',
        transform=val_transform,
    )
    testset_target = dalib.vision.datasets.Office31(
        root='./data/',
        task='W',
        transform=val_transform,
    )
    return trainset_source, trainset_target, testset_source, testset_target