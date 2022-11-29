from utils.ood_detection.load_data import get_cifar10_test_dataloader, get_cifar100_test_dataloader, \
    get_SVHN_test_dataloader, get_mixed_dataloader, get_cifar10_train_valid_dataloader, \
    get_cifar100_train_valid_dataloader, get_SVHN_train_valid_dataloader


def get_mixed_test_dataloader(args):
    if args.data_id.lower() == "cifar10":
        ID_test_dataloader = get_cifar10_test_dataloader(args)  # get ID
    elif args.data_id.lower() == "cifar100":
        ID_test_dataloader = get_cifar100_test_dataloader(args)
    elif args.data_id.lower() == "svhn":
        ID_test_dataloader = get_SVHN_test_dataloader(args)
    else:
        raise ValueError('Wrong ID dataset specified in args.')

    if args.data_ood.lower() == "cifar10":
        OOD_test_dataloader = get_cifar10_test_dataloader(args)  # get OOD
    elif args.data_ood.lower() == "cifar100":
        OOD_test_dataloader = get_cifar100_test_dataloader(args)
    elif args.data_ood.lower() == "svhn":
        OOD_test_dataloader = get_SVHN_test_dataloader(args)
    else:
        raise ValueError('Wrong OOD dataset specified in args.')

    return get_mixed_dataloader(args, ID_test_dataloader.dataset, OOD_test_dataloader.dataset)


def get_mixed_train_valid_dataloaders(args):
    if args.data_id.lower() == "cifar10":
        ID_train_dataloader, ID_valid_dataloader = get_cifar10_train_valid_dataloader(args)  # get ID
    elif args.data_id.lower() == "cifar100":
        ID_train_dataloader, ID_valid_dataloader = get_cifar100_train_valid_dataloader(args)
    elif args.data_id.lower() == "svhn":
        ID_train_dataloader, ID_valid_dataloader = get_SVHN_train_valid_dataloader(args)
    else:
        raise ValueError('Wrong ID dataset specified in args.')

    if args.data_ood.lower() == "cifar10":
        OOD_train_dataloader, OOD_valid_dataloader = get_cifar10_train_valid_dataloader(args)  # get OOD
    elif args.data_ood.lower() == "cifar100":
        OOD_train_dataloader, OOD_valid_dataloader = get_cifar100_train_valid_dataloader(args)
    elif args.data_ood.lower() == "svhn":
        OOD_train_dataloader, OOD_valid_dataloader = get_SVHN_train_valid_dataloader(args)
    else:
        raise ValueError('Wrong OOD dataset specified in args.')

    mixed_train_dataloader = get_mixed_dataloader(args, ID_train_dataloader.dataset,
                                                   OOD_train_dataloader.dataset)
    mixed_valid_dataloader = get_mixed_dataloader(args, ID_valid_dataloader.dataset,
                                                   OOD_valid_dataloader.dataset)

    return mixed_train_dataloader, mixed_valid_dataloader


def get_test_dataloader(args):
    if args.dataset.lower() == "cifar10":
        test_dataloader = get_cifar10_test_dataloader(args)  # get ID
    elif args.dataset.lower() == "cifar100":
        test_dataloader = get_cifar100_test_dataloader(args)
    elif args.dataset.lower() == "svhn":
        test_dataloader = get_SVHN_test_dataloader(args)
    else:
        raise ValueError('Wrong ID dataset specified in args.')

    return test_dataloader


def get_train_valid_dataloaders(args):
    if args.dataset.lower() == "cifar10":
        train_dataloader, valid_dataloader = get_cifar10_train_valid_dataloader(args)  # get ID
    elif args.dataset.lower() == "cifar100":
        train_dataloader, valid_dataloader = get_cifar100_train_valid_dataloader(args)
    elif args.dataset.lower() == "svhn":
        train_dataloader, valid_dataloader = get_SVHN_train_valid_dataloader(args)
    else:
        raise ValueError('Wrong dataset specified in args.')

    return train_dataloader, valid_dataloader
