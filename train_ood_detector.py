from tqdm import tqdm
import time
import torch
import torchvision

from utils.dotdict import dotdict
from utils.normalize_image_data import normalize_cifar100_image_data, normalize_cifar10_image_data,\
    normalize_general_image_data
from utils.ood_detection.load_data import get_mixed_dataloader, shuffle_batch_elements, get_cifar10_train_valid_dataloader, \
    get_cifar10_test_dataloader, get_cifar100_train_valid_dataloader, get_cifar100_test_dataloader,\
    get_SVHN_train_valid_dataloader, get_SVHN_test_dataloader
from utils.models.ood_detector import MiniNet, CNN_IBP
from vit.src.model import VisionTransformer as ViT


def train_detector(args):
    model = get_model_from_args(args)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    mixed_train_dataloader, mixed_valid_dataloader = get_mixed_train_valid_dataloaders(args)

    epoch_number = 0
    best_vloss = 1_000_000.
    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(args, epoch_number, model, loss_fn, optimizer, mixed_train_dataloader)

        # We don't need gradients on to do reporting
        model.train(False)

        with torch.no_grad():
            #Error
            running_valid_error = 0
            running_vloss = 0.0
            for i, (vdata_id, vdata_ood) in enumerate(mixed_valid_dataloader):
                vinputs, vlabels = shuffle_batch_elements(vdata_id, vdata_ood)
                vinputs, vlabels = vinputs.to(device=args.device), vlabels.to(device=args.device)
                vinputs.requires_grad = True  # possible because leaf of the acyclic graph
                vinputs = normalize_general_image_data(vinputs)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs.squeeze(1), vlabels)
                running_vloss += vloss
                running_valid_error += error_criterion(voutputs.squeeze(1), vlabels)

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            #Error
            avg_valid_error = running_valid_error / (i+1)
            print("Average Validation Error:", avg_valid_error.item())

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                # TODO comment out if models should be saved
                # model_path = 'model_{}_{}'.format(epoch_number)
                # torch.save(model.state_dict(), model_path)

        epoch_number += 1

    if args.save_model:
        # Save the model
        model_path = "utils/models/saved_models/"
        saved_model_name = args.model_name + "_" + str(args.img_size) + "SupCE_ID" + args.data_id + "_OOD"\
                           + args.data_ood + "_bs" + str(args.batch_size) + "_lr" + str(args.lr).strip(".") + "_epochs"\
                           + str(args.epochs) + "_" + str(int(time.time())) + ".pth"
        torch.save(model.state_dict(), model_path+saved_model_name)
        print("Model saved in path: ", model_path+saved_model_name)


def train_one_epoch(args, epoch_index, model, loss_fn, optimizer, mixed_train_dataloader):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (data_id, data_ood) in enumerate(tqdm(mixed_train_dataloader)):
        # Every data instance is an input + label pair
        inputs, labels = shuffle_batch_elements(data_id, data_ood)
        inputs, labels = inputs.to(device=args.device), labels.to(device=args.device)
        inputs.requires_grad = True #possible because leaf of the acyclic graph
        # labels.requires_grad = True
        inputs = normalize_general_image_data(inputs)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(1), labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(mixed_train_dataloader) + i + 1
            running_loss = 0.

    return last_loss


def error_criterion(outputs,labels):
    """
    used to calculate the errors in the validation phase
    """
    prediction_tensor = torch.where(outputs>0.,1.,0.)
    train_error = (prediction_tensor != labels).float().sum()/prediction_tensor.size()[0]
    return train_error


def get_model_from_args(args):
    if args.model_name.lower() == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=1).to(device=args.device) # cuda()
    elif args.model_name.lower() == "vit":
        model = ViT(image_size=(args.img_size, args.img_size),  # 224,224
                    num_heads=args.num_heads, #12 #also a very small amount of heads to speed up training
                    num_layers=args.num_layers,  # 12 # 5 is a very small vit model
                    num_classes=1,  # one is enough to detect two classes, ID and OOD
                    contrastive=False,
                    timm=True).to(device=args.device) # cuda()
    elif args.model_name.lower() == "cnn_inp":
        model = CNN_IBP().to(device=args.device)
    elif args.model_name.lower() == "mininet":
        model = MiniNet().to(device=args.device)
    else:
        print("Error - wrong model specified in 'super_args'")
        import sys
        sys.exit()
    return model

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


def test_detector(args):
    model_path = "utils/models/saved_models/"
    #saved_model_name = args.model_name + "_" + str(args.img_size) + "SupCE_ID" + args.data_id + "_OOD"\
    #                       + args.data_ood + "_bs" + str(args.batch_size) + "_lr" + str(args.lr).strip(".") + "_epochs"\
    #                       + str(args.epochs) + "_" + str(int(time.time())) + ".pth"

    saved_model_name = "resnet_224SupCE_IDcifar10_OODsvhn_bs32_lr0.0001_epochs8_1667912039.pth"

    # Maybe some adjustments in the 'if' condition, if not running with the args
    if args.model_name.lower() == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=1).to(device=args.device)  # cuda()
    elif args.model_name.lower() == "vit":
        model = ViT(image_size=(args.img_size, args.img_size),  # 224,224
                    num_heads=args.num_heads,  # 12 #also a very small amount of heads to speed up training
                    num_layers=args.num_layers,  # 12 # 5 is a very small vit model
                    num_classes=1,  # one is enough to detect two classes, ID and OOD
                    contrastive=False,
                    timm=True).to(device=args.device)  # cuda()
    elif args.model_name.lower() == "cnn_inp":
        model = CNN_IBP().to(device=args.device)
    elif args.model_name.lower() == "mininet":
        model = MiniNet().to(device=args.device)
    else:
        raise ValueError('Model type not supported for testing.')

    model.load_state_dict(torch.load(model_path+saved_model_name))
    # get a dataloader mixed 50:50 with ID and OOD data and labels of 0 (ID) and 1 (OOD)
    mixed_test_dataloader = get_mixed_test_dataloader(args)

    print("Start Testing")
    with torch.no_grad():
        model.eval()
        running_test_error = 0
        for i, (data_id, data_ood) in enumerate(tqdm(mixed_test_dataloader)):
            inputs, labels = shuffle_batch_elements(data_id, data_ood)
            inputs, labels = inputs.to(device=args.device), labels.to(device=args.device)
            inputs = normalize_general_image_data(inputs)
            outputs = model(inputs)
            running_test_error += error_criterion(outputs.squeeze(1), labels)

        # Error
        avg_valid_error = running_test_error / (i + 1)
        print("Average Test Error:", avg_valid_error.item())
        print("Finished Testing the Model")


if __name__ == '__main__':

    #model_name = input('Enter the model name ("resnet", "vit", "mininet", "cnn_ibp"):')

    super_args = dotdict({
        "train": False,
        "test": True,
        "save_model": True,

        "device": "cuda",
        "workers": 0,
        "model_name": "resnet", #model_name, # "resnet", "vit", "mininet", "cnn_ibp"
        "num_heads": 4,
        "num_layers": 5,

        "epochs": 8,
        "lr": 0.0001,  # 0.001 = 1e-3   # 0.0001 = 1e-4
        "momentum": 0.9, # 0.9,

        "batch_size": 32, #16,
        "img_size": 224,
        "data_id": "cifar10", # "cifar10", "cifar100", "svhn"
        "data_ood": "svhn"
    })

    if super_args.train:
        train_detector(super_args)

    if super_args.test:
        test_detector(super_args)

    print("finish all execution")

    # TODO
    # 1. set all labels of cifar10 to 0 (for ID) and cifar100 to 1 (for OOD)
    # 2. throw them into a model (CNN, ResNet, ViT)
    # 3. Train a model with the new labels
    # 3.a) check if all models work with training, saving, loading & testing
    # 4. make perturbations on them and see if the model still detects them correctly
    # 5. switch out the models and train another model


