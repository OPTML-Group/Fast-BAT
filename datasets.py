import glob
import os
from shutil import move

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, STL10, CelebA, ImageFolder

__all__ = ['mnist_dataloader', 'cifar10_dataloader', 'cifar100_dataloader', 'tiny_imagenet_dataloader',
           'svhn_dataloader', 'stl10_dataloader', 'celeba_dataloader', 'imagenet_dataloader', 'gtsrb_dataloader']


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def imagenet_dataloader(batch_size=128, data_dir='/data'):
    data_dir = os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC')
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    resize_transform = []
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(resize_transform + [
            transforms.RandomResizedCrop(288),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(resize_transform + [
            transforms.CenterCrop(288),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(mean=torch.tensor([0.485, 0.456, 0.406]),
                                                      std=torch.tensor([0.229, 0.224, 0.225]))

    num_classes = 1000

    return train_loader, val_loader, val_loader, dataset_normalization, num_classes


def celeba_dataloader(batch_size=64, data_dir='./data'):
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = CelebA(data_dir, "train", transform=train_transform, download=True)
    val_set = CelebA(data_dir, "valid", transform=test_transform, download=True)
    test_set = CelebA(data_dir, "test", transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    assert NotImplementedError("Not Ready for Use!")

    return train_loader, val_loader, test_loader, None


def mnist_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.1):
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_size = int(60000 * (1 - val_ratio))
    val_size = 60000 - train_size

    train_set = Subset(MNIST(data_dir, train=True, transform=train_transform, download=True), list(range(train_size)))
    val_set = Subset(MNIST(data_dir, train=True, transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = MNIST(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = 10

    return train_loader, val_loader, test_loader, num_classes


def cifar10_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.1):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_size = int(50000 * (1 - val_ratio))
    val_size = 50000 - train_size

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(train_size)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    num_classes = 10
    return train_loader, val_loader, test_loader, dataset_normalization, num_classes


def cifar100_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.1):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_size = int(50000 * (1 - val_ratio))
    val_size = 50000 - train_size

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True),
                       list(range(train_size)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    num_classes = 100

    return train_loader, val_loader, test_loader, dataset_normalization, num_classes


def tiny_imagenet_dataloader(batch_size=64, data_dir='./data/tiny_imagenet/', permutation_seed=10):
    """
    Prepare for the Tiny-ImageNet dataset
    Step 1: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    Step 2: unzip -qq 'tiny-imagenet-200.zip'
    Step 3: rm tiny-imagenet-200.zip (optional)
    Code primarily from https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
    Args:
        batch_size:
        data_dir:
        permutation_seed:

    Returns:

    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train/')
    val_path = os.path.join(data_dir, 'val/')
    test_path = os.path.join(data_dir, 'test/')

    if os.path.exists(os.path.join(val_path, "images")):
        if os.path.exists(test_path):
            os.rename(test_path, os.path.join(data_dir, "test_original"))
            os.mkdir(test_path)
        val_dict = {}
        val_anno_path = os.path.join(val_path, "val_annotations.txt")
        with open(val_anno_path, 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]

        paths = glob.glob('./tiny-imagenet-200/val/images/*')
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(val_path + str(folder)):
                os.mkdir(val_path + str(folder))
                os.mkdir(val_path + str(folder) + '/images')
            if not os.path.exists(test_path + str(folder)):
                os.mkdir(test_path + str(folder))
                os.mkdir(test_path + str(folder) + '/images')

        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if len(glob.glob(val_path + str(folder) + '/images/*')) < 25:
                dest = val_path + str(folder) + '/images/' + str(file)
            else:
                dest = test_path + str(folder) + '/images/' + str(file)
            move(path, dest)

        os.rmdir(os.path.join(val_path, "images"))

    np.random.seed(permutation_seed)

    train_set = Subset(ImageFolder(train_path, transform=train_transform), range(100000))
    val_set = Subset(ImageFolder(val_path, transform=test_transform), range(10000))
    test_set = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    num_classes = 200

    return train_loader, val_loader, test_loader, dataset_normalization, num_classes


def svhn_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.1):
    num_workers = 2
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_size = int(73257 * (1 - val_ratio))
    val_size = 73257 - train_size

    train_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True), list(range(train_size)))
    val_set = Subset(SVHN(data_dir, split='train', transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                             drop_last=True)
    dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])

    num_classes = 10

    return train_loader, val_loader, test_loader, dataset_normalization, num_classes


def stl10_dataloader(batch_size=64, data_dir='./data/'):
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = STL10(data_dir, split='train', download=True, transform=train_transform)
    test_set = STL10(data_dir, split='test', download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4467, 0.4398, 0.4066], std=[0.2242, 0.2215, 0.2239])

    num_classes = 10
    return train_loader, test_loader, test_loader, dataset_normalization, num_classes


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):
        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        print("Reading GTSRB data......")
        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

        self.imgs = []
        self.labels = []

        print("Processing GTSRB data......")
        for idx in range(len(self.csv_data)):
            img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                    self.csv_data.iloc[idx, 0])
            img = Image.open(img_path)
            classId = self.csv_data.iloc[idx, 1]
            self.labels.append(classId)

            if self.transform is not None:
                img = self.transform(img)

            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


def gtsrb_dataloader(batch_size=128, data_dir='./data/', val_ratio=0.1):
    """
    Code Ref: https://github.com/tomlawrenceuk/GTSRB-Dataloader/blob/master/gtsrb_dataset.py
    Download dataset from https://onedrive.live.com/?authkey=%21AKNpIXu0xpmVm1I&cid=25B382439BAD237F&id=25B382439BAD237F%21224763&parId=25B382439BAD237F%21224762&action=locate
    Unzip the zip file and make the path the data_dir below.

    Args:
        data_dir: see ABOVE
    Returns:

    """
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    number_train_images = 39208

    train_size = int(number_train_images * (1 - val_ratio))
    val_size = number_train_images - train_size
    train_set = Subset(GTSRB(data_dir, train=True, transform=train_transform), list(range(train_size)))
    val_set = Subset(GTSRB(data_dir, train=True, transform=test_transform),
                     list(range(train_size, train_size + val_size)))
    test_set = GTSRB(data_dir, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    num_classes = 43
    return train_loader, val_loader, test_loader, dataset_normalization, num_classes
