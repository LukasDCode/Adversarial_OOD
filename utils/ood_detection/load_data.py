import random
import torch
import numpy as np

from utils.dotdict import dotdict
from vit.src.data_loaders import get_transform

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import SVHN


def get_cifar10_train_valid_dataloader(args):
    """
    dataloaders_config = {
        "data_dir": "/home/wiss/koner/lukas/Adversarial_OOD/data/cifar10/cifar-10-batches-py/",
        "image_size": args.img_size,  # 224
        "batch_size": int(args.batch_size/2),  # 16/2 = 8
        "num_workers": args.workers,  # 0
        "contrastive": False,
        "albumentation": False,
        "net": "vit",  # does not even get used inside create_dataloader function
        "no_train_aug": False,
        "dataset": "cifar10",  # "cifar100"
        "deit": False
    }
    dataloaders_config = dotdict(dataloaders_config)
    train_dataloader, test_dataloader = create_dataloaders(dataloaders_config)
    return train_dataloader, test_dataloader
    """
    transform = get_transform(train=False, image_size=args.img_size, dataset='cifar10', network=args.model_name)
    dataset_train = datasets.CIFAR10(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar10/', train=True, download=True, transform=transform)
    dataset_train, dataset_valid = split_test_valid_dataset(dataset_train)
    return create_dataloader_from_dataset(args, dataset_train), create_dataloader_from_dataset(args, dataset_valid)


def get_cifar10_test_dataloader(args):
    transform = get_transform(train=False, image_size=args.img_size, dataset='cifar10', network=args.model_name)
    dataset_test = datasets.CIFAR10(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar10/', train=False, download=True, transform=transform)
    return create_dataloader_from_dataset(args, dataset_test)


def get_cifar100_train_valid_dataloader(args):
    """
    dataloaders_config = {
        "data_dir": "/home/wiss/koner/lukas/Adversarial_OOD/data/cifar100/cifar-100-python/",
        # "/home/koner/adversarial_ood/data/cifar-100-python/"
        "image_size": args.img_size,  # 224
        "batch_size": int(args.batch_size/2),  # 16/2 = 8
        "num_workers": args.workers,  # 0
        "contrastive": False,
        "albumentation": False,
        "net": "resnet", #resnet # does not even get used inside create_dataloader function
        "no_train_aug": False,
        "dataset": "cifar100",  # "cifar100"
        "deit": False
    }
    dataloaders_config = dotdict(dataloaders_config)
    train_dataloader, test_dataloader = create_dataloaders(dataloaders_config)
    return train_dataloader, test_dataloader
    """

    transform = get_transform(train=True, image_size=args.img_size, dataset='cifar100', network=args.model_name)
    dataset_train = datasets.CIFAR100(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar100/', train=True, download=True, transform=transform)
    dataset_train, dataset_valid = split_test_valid_dataset(dataset_train)
    return create_dataloader_from_dataset(args, dataset_train), create_dataloader_from_dataset(args, dataset_valid)
    # '/home-local/koner/lukas/Adversarial_OOD/data/cifar100/cifar-100-python/'


def get_cifar100_test_dataloader(args):
    transform = get_transform(train=False, image_size=args.img_size, dataset='cifar100', network=args.model_name)
    dataset_test = datasets.CIFAR100(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar100/', train=False, download=True, transform=transform)
    return create_dataloader_from_dataset(args, dataset_test)


def get_SVHN_train_valid_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), )
    ])
    dataset_train = SVHN(root="/home/wiss/koner/lukas/Adversarial_OOD/data/svhn/", split="train", download=True, transform=transform)
    dataset_train, dataset_valid = split_test_valid_dataset(dataset_train)
    #dataset_test = SVHN(root="/home/wiss/koner/lukas/Adversarial_OOD/data/svhn/", split="test", download=True, transform=transform)

    return create_dataloader_from_dataset(args, dataset_train), create_dataloader_from_dataset(args, dataset_valid)


def get_SVHN_test_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), )
    ])
    dataset_test = SVHN(root="/home/wiss/koner/lukas/Adversarial_OOD/data/svhn/", split="test", download=True, transform=transform)

    return create_dataloader_from_dataset(args, dataset_test)


def split_test_valid_dataset(dataset_train):
    # returns train_dataset and valid_dataset split of 90% to 10%
    train_length = int(len(dataset_train) * 0.9)
    return random_split(dataset_train, [train_length, len(dataset_train)-train_length])

def create_dataloader_from_dataset(args, dataset, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=int(args.batch_size / 2), shuffle=shuffle, num_workers=args.workers, pin_memory=True)
    #valid_loader = DataLoader(dataset_valid, batch_size=int(args.batch_size / 2), shuffle=True, num_workers=args.workers, pin_memory=True)
    #test_loader = DataLoader(dataset_test, batch_size=int(args.batch_size / 2), num_workers=args.workers, pin_memory=True)
    return dataloader


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
        #return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_mixed_dataloader(args, dataset_id, dataset_ood):
    """
    combines the ID and the OOD dataset into one single dataloader
    """
    return torch.utils.data.DataLoader(
                ConcatDataset(dataset_id, dataset_ood),
                batch_size=int(args.batch_size/2), # 16/2=8
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
            )


def shuffle_batch_elements(data_id, data_ood):
    """
    shuffles the samples of ID and OOD data batch --> return: shuffled_inputs
    also shuffles the labels of ID and OOD of the same data batch in the same order --> return: shuffled_targets
    in the end every shuffled_inputs shuffled_targets tensor contains batch_size/2 samples of the ID and
    batch_size/2 samples of the OOD dataset (default is 8 ID + 8 OOD)
    """
    randomness = list(range(data_id[1].size(dim=0) + data_ood[1].size(dim=0)))  # list of indices will determine the shuffle of inputs & targets alike
    random.shuffle(randomness)  # shuffle tensor elements randomly

    # some dataloaders return a list within a list with duplicates, but we only need one of those doubles
    if type(data_id[0]) == list:
        data_id[0] = data_id[0][0]

    shuffled_inputs = torch.cat((data_id[0], data_ood[0]), 0)[randomness]
    shuffled_targets = torch.cat((torch.zeros(data_id[1].size(dim=0)), torch.ones(data_ood[1].size(dim=0))), 0)[randomness]

    return shuffled_inputs, shuffled_targets
