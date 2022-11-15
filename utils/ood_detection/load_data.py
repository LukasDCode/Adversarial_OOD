import random
import torch

from vit.src.data_loaders import get_transform

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import SVHN


def get_cifar10_train_valid_dataloader(args):
    if args.detector_model_name:
        model_name = args.detector_model_name
    else:
        model_name = args.classification_model_name
    transform = get_transform(train=False, image_size=args.img_size, dataset='cifar10', network=model_name)
    dataset_train = datasets.CIFAR10(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar10/', train=True, download=True, transform=transform)
    dataset_train, dataset_valid = split_test_valid_dataset(dataset_train)
    return create_dataloader_from_dataset(args, dataset_train), create_dataloader_from_dataset(args, dataset_valid)


def get_cifar10_test_dataloader(args):
    if args.detector_model_name:
        model_name = args.detector_model_name
    else:
        model_name = args.classification_model_name
    transform = get_transform(train=False, image_size=args.img_size, dataset='cifar10', network=model_name)
    dataset_test = datasets.CIFAR10(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar10/', train=False, download=True, transform=transform)
    return create_dataloader_from_dataset(args, dataset_test)


def get_cifar100_train_valid_dataloader(args):
    if args.detector_model_name:
        model_name = args.detector_model_name
    else:
        model_name = args.classification_model_name
    transform = get_transform(train=True, image_size=args.img_size, dataset='cifar100', network=model_name)
    dataset_train = datasets.CIFAR100(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar100/', train=True, download=True, transform=transform)
    dataset_train, dataset_valid = split_test_valid_dataset(dataset_train)
    return create_dataloader_from_dataset(args, dataset_train), create_dataloader_from_dataset(args, dataset_valid)
    # '/home-local/koner/lukas/Adversarial_OOD/data/cifar100/cifar-100-python/'


def get_cifar100_test_dataloader(args):
    if args.detector_model_name:
        model_name = args.detector_model_name
    else:
        model_name = args.classification_model_name
    transform = get_transform(train=False, image_size=args.img_size, dataset='cifar100', network=model_name)
    dataset_test = datasets.CIFAR100(root='/home-local/koner/lukas/Adversarial_OOD/data/cifar100/', train=False, download=True, transform=transform)
    return create_dataloader_from_dataset(args, dataset_test)


def get_SVHN_train_valid_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), )
    ])
    # /home/wiss/koner/lukas/Adversarial_OOD/data/svhn/
    dataset_train = SVHN(root="/home-local/koner/lukas/Adversarial_OOD/data/svhn/", split="train", download=True, transform=transform)
    dataset_train, dataset_valid = split_test_valid_dataset(dataset_train)
    #dataset_test = SVHN(root="/home/wiss/koner/lukas/Adversarial_OOD/data/svhn/", split="test", download=True, transform=transform)

    return create_dataloader_from_dataset(args, dataset_train), create_dataloader_from_dataset(args, dataset_valid)


def get_SVHN_test_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), )
    ])
    dataset_test = SVHN(root="/home-local/koner/lukas/Adversarial_OOD/data/svhn/", split="test", download=True, transform=transform)

    return create_dataloader_from_dataset(args, dataset_test)


def split_test_valid_dataset(dataset_train):
    # returns train_dataset and valid_dataset split of 90% to 10%
    train_length = int(len(dataset_train) * 0.9)
    return random_split(dataset_train, [train_length, len(dataset_train)-train_length])

def create_dataloader_from_dataset(args, dataset, shuffle=True):
    return DataLoader(dataset, batch_size=int(args.batch_size / 2), shuffle=shuffle, num_workers=args.workers, pin_memory=True)


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

    """
    # some dataloaders return a list within a list with duplicates, but we only need one of those doubles
    if type(data_id[0]) == list:
        print("Data ID list thing happened again +++++")
        data_id[0] = data_id[0][0]
    """

    shuffled_inputs = torch.cat((data_id[0], data_ood[0]), 0)[randomness]
    shuffled_targets = torch.cat((torch.zeros(data_id[1].size(dim=0)), torch.ones(data_ood[1].size(dim=0))), 0)[randomness]
    shuffled_targets = shuffled_targets.unsqueeze(1).repeat(1, 2)  # repeats the labels in the 2nd dimension of the tensor
    # if the duplicated labels tensor doesnt work, maybe flipping the 2nd dimension with a Tile '~' works
    # shuffled_targets[1] = ~shuffled_targets[1] or something like this

    del data_id[1], data_ood[1] # remove the unused original class labels for performance (not 