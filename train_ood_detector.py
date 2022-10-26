import random
from tqdm import tqdm
import torch
import torchvision

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.dotdict import dotdict
from vit.src.data_loaders import create_dataloaders
from utils.models.ood_detector import GarmentClassifier


def get_cifar10_dataloader(args):
    dataloaders_config = {
        "data_dir": "/home/wiss/koner/lukas/Adversarial_OOD/data/cifar10/cifar-10-batches-py/",
        "image_size": 224,  # 224
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
    train_dataloader, valid_dataloader = create_dataloaders(dataloaders_config)
    return train_dataloader, valid_dataloader

def get_cifar100_dataloader(args):
    dataloaders_config = {
        "data_dir": "/home/wiss/koner/lukas/Adversarial_OOD/data/cifar100/cifar-100-python/",
        # "/home/koner/adversarial_ood/data/cifar-100-python/"
        "image_size": 224,  # 224
        "batch_size": int(args.batch_size/2),  # 16/2 = 8
        "num_workers": args.workers,  # 0
        "contrastive": False,
        "albumentation": False,
        "net": "vit",  # does not even get used inside create_dataloader function
        "no_train_aug": False,
        "dataset": "cifar100",  # "cifar100"
        "deit": False
    }
    dataloaders_config = dotdict(dataloaders_config)
    train_dataloader, valid_dataloader = create_dataloaders(dataloaders_config)
    return train_dataloader, valid_dataloader

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
        #return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def get_mixed_data_loader(args, dataset_id, dataset_ood):
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
    randomness = list(range(16))  # indices will determine the shuffle of inputs & targets alike
    random.shuffle(randomness)  # shuffle tensor elements randomly

    shuffled_inputs = torch.cat((data_id[0], data_ood[0]), 0)[randomness]
    shuffled_targets = torch.cat((torch.zeros(data_id[1].size(dim=0)), torch.ones(data_ood[1].size(dim=0))), 0).type(torch.LongTensor)[randomness]

    return shuffled_inputs, shuffled_targets






def train_detector(args, model):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    args.loss_fn = loss_fn
    args.optimizer = optimizer

    epoch_number = 0
    best_vloss = 1_000_000.
    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        cifar10_train_dataloader, cifar10_valid_dataloader = get_cifar10_dataloader(args)  # get ID
        cifar100_train_dataloader, cifar100_valid_dataloader = get_cifar100_dataloader(args)  # get OOD

        mixed_train_data_loader = get_mixed_data_loader(args, cifar10_train_dataloader.dataset,
                                                        cifar100_train_dataloader.dataset)
        mixed_valid_data_loader = get_mixed_data_loader(args, cifar10_valid_dataloader.dataset,
                                                        cifar100_valid_dataloader.dataset)


        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(args, epoch_number, mixed_train_data_loader, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, (vdata_id, vdata_ood) in enumerate(mixed_valid_data_loader):
            vinputs, vlabels = shuffle_batch_elements(vdata_id, vdata_ood)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

def train_one_epoch(args, epoch_index, mixed_train_data_loader, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (data_id, data_ood) in enumerate(tqdm(mixed_train_data_loader)):
        # Every data instance is an input + label pair
        inputs, labels = shuffle_batch_elements(data_id, data_ood)

        # Zero your gradients for every batch!
        args.optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = args.loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        args.optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(mixed_train_data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss




if __name__ == '__main__':

    super_args = dotdict({
        "batch_size": 16,
        "workers": 0,
        "epochs": 1,
        "lr": 0.001,
        "momentum": 0.9,
        "model": "resnet"
    })

    if super_args.model.lower() == "resnet":
        model = torchvision.models.resnet18(pretrained=False)
    elif super_args.model.lower() == "vit":
        # TODO get VIT model here
        model = "nothing"
    else:
        print("Error - wrong model specified in 'super_args'")
        import sys
        sys.exit()

    train_detector(super_args, model)

    print("finish")

    # TODO
    # 1. set all labels of cifar10 to 0 (for ID) and cifar100 to 1 (for OOD)
    # 2. throw them into a model (CNN, ResNet, ViT)
    # 3. Train a model with the new labels
    # 4. make perturbations on them and see if the model still detects them correctly
    # 5. switch out the models and train another model

