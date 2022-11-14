import argparse
import time
from tqdm import tqdm
import torch

from utils.dotdict import dotdict
from utils.normalize_image_data import normalize_general_image_data, normalize_cifar10_image_data, normalize_cifar100_image_data, normalize_SVHN_image_data
from train_ood_detector import get_model_from_args
from ood_detection.load_data import get_train_valid_dataloaders, get_test_dataloader


def train_classifier(args):
    if args.device == "cuda": torch.cuda.empty_cache()

    classification_model = get_model_from_args(args, args.classification_model_name, num_classes=args.num_classes)
    # TODO different loss for vit model
    if args.loss == "ce":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        print("Error - specified loss not available, currently only working with CrossEntropy loss")
        return
    optimizer = torch.optim.SGD(classification_model.parameters(), lr=args.lr, momentum=args.momentum)

    train_dataloader, valid_dataloader = get_train_valid_dataloaders(args)

    epoch_number = 0
    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        classification_model.train(True)
        # TODO train one epoch normally and one epoch with adversarially altered images
        avg_loss = train_one_epoch(args, classification_model, loss_fn, optimizer, train_dataloader)
        # We don't need gradients on to do reporting
        classification_model.train(False)  # same as model.eval()

        with torch.no_grad():
            # Error calculation
            running_valid_error = 0
            running_vloss = 0.0
            for i, (vinputs, vlabels) in enumerate(valid_dataloader):
                vinputs, vlabels = vinputs.to(device=args.device), vlabels.to(device=args.device)
                if args.dataset.lower() == "cifar10":
                    normalized_vinputs = normalize_cifar10_image_data(vinputs)
                elif args.dataset.lower() == "cifar100":
                    normalized_vinputs = normalize_cifar100_image_data(vinputs)
                elif args.dataset.lower() == "svhn":
                    normalized_vinputs = normalize_SVHN_image_data(vinputs)
                else:
                    normalized_vinputs = normalize_general_image_data(vinputs)
                voutputs = classification_model(normalized_vinputs)
                # voutputs: [batch_size, num_classes]
                # vlabels:  [batch_size]
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
                running_valid_error += error_criterion(voutputs, vlabels)
                if args.device == "cuda": torch.cuda.empty_cache()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Error
            avg_valid_error = running_valid_error / (i + 1)
            print("Average Validation Error:", avg_valid_error.item())

        epoch_number += 1

    if args.save_model:
        # Save the model
        model_path = "utils/models/saved_models/"
        saved_model_name = args.classification_model_name + "_" + str(args.img_size) + "SupCE_" + args.dataset + "_bs"\
                           + str(args.batch_size/2) + "_lr" + str(args.lr).strip(".") + "_epochs" + str(args.epochs)\
                           + "_" + str(int(time.time())) + ".pth"

        saved_model_name = "test.pth"

        #torch.save(classification_model.state_dict(), model_path + saved_model_name)
        torch.save({
                'model_name': args.classification_model_name,
                'img_size': args.img_size,
                'dataset': args.dataset,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'epoch': args.epochs,
                'model_state_dict': classification_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': args.loss,
            }, model_path + saved_model_name)
        print("Model saved in path: ", model_path + saved_model_name)


def train_one_epoch(args, classification_model, loss_fn, optimizer, train_dataloader):
    running_loss, last_loss = 0., 0.

    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        inputs, labels = inputs.to(device=args.device), labels.to(device=args.device)
        inputs.requires_grad = True  # possible because leaf of the acyclic graph
        if args.dataset.lower() == "cifar10":
            normalized_inputs = normalize_cifar10_image_data(inputs)
        elif args.dataset.lower() == "cifar100":
            normalized_inputs = normalize_cifar100_image_data(inputs)
        elif args.dataset.lower() == "svhn":
            normalized_inputs = normalize_SVHN_image_data(inputs)
        else:
            normalized_inputs = normalize_general_image_data(inputs)
        # inputs:              [batch_size, channels, img_size, img_size]
        # normalized_inputs:   [batch_size, channels, img_size, img_size]

        optimizer.zero_grad()  # Zero gradients for every batch
        outputs = classification_model(normalized_inputs)
        # outputs:  [batch_size, num_classes]
        # labels:   [batch_size]
        loss = loss_fn(outputs, labels)  # Compute the loss and its gradients
        loss.backward()
        optimizer.step()  # Adjust learning weights
        running_loss += loss.item()  # Gather data and report
        del normalized_inputs, loss  # delete for performance reasons to free up cuda memory
        if args.device == "cuda": torch.cuda.empty_cache()

        print_interval = 100
        if i % print_interval == print_interval - 1:
            last_loss = running_loss / print_interval  # loss per batch
            print("   Batch", i + 1, "loss:", last_loss)
            running_loss = 0.

    return last_loss


def test_classifier(args):
    if args.classification_ckpt:
        checkpoint_path = args.classification.ckpt
    else:
        checkpoint_path = "utils/models/saved_models/test.pth"
        # "/nfs/data3/koner/contrastive_ood/save/vit/vit_224SupCE_cifar10_bs512_lr0.01_wd1e-05_temp_0.1_210316_122535/checkpoints/ckpt_epoch_50.pth"

    checkpoint = torch.load(checkpoint_path)
    args.epochs = checkpoint['epoch']
    args.loss = checkpoint['loss']
    args.classification_model_name = checkpoint['model_name']
    args.img_size = checkpoint['img_size']
    args.dataset = checkpoint['dataset']
    args.batch_size = checkpoint['batch_size']
    args.lr = checkpoint['lr']

    model = get_model_from_args(args, args.classification_model_name, args.num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])


    # get a dataloader mixed 50:50 with ID and OOD data and labels of 0 (ID) and 1 (OOD)
    test_dataloader = get_test_dataloader(args)

    print("Start Testing")
    with torch.no_grad():
        model.eval()
        running_test_error = 0
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.to(device=args.device), labels.to(device=args.device)
            if args.dataset.lower() == "cifar10":
                normalized_inputs = normalize_cifar10_image_data(inputs)
            elif args.dataset.lower() == "cifar100":
                normalized_inputs = normalize_cifar100_image_data(inputs)
            elif args.dataset.lower() == "svhn":
                normalized_inputs = normalize_SVHN_image_data(inputs)
            else:
                normalized_inputs = normalize_general_image_data(inputs)  # no detach because requires gradient
            outputs = model(normalized_inputs)
            running_test_error += error_criterion(outputs.squeeze(1), labels)
            if args.device == "cuda": torch.cuda.empty_cache()

        # Error
        avg_valid_error = running_test_error / (i + 1)
        print("Average Test Error:", avg_valid_error.item())
        print("Finished Testing the Model")


def error_criterion(outputs, labels):
    """
    used to calculate the errors in the validation phase
    """
    prediction_tensor = torch.max(outputs, dim=1) #torch.where(outputs > 0., 1., 0.)
    train_error = (prediction_tensor.indices != labels).float().sum() / prediction_tensor.indices.size()[0]
    return train_error


def parse_args():
    parser = argparse.ArgumentParser(description='Run the monotone PGD attack on a batch of images, default is with ViT and the MPGD of Alex, where cifar10 is ID and cifar100 is OOD')

    parser.add_argument('--classification_model_name', type=str, default="vit", help='str - what model should be used to classify input samples "vit", "resnet", "mininet" or "cnn_ibp"')
    parser.add_argument('--classification_ckpt', type=str, default=None, help='str - path of pretrained model checkpoint')
    parser.add_argument('--device', type=str, default="cuda", help='str - cpu or cuda to calculate the tensors on')
    parser.add_argument('--dataset', type=str, default="cifar10", help='str - the in-distribution dataset "cifar10", "cifar100" or "svhn"')
    parser.add_argument('--num_classes', type=int, default=10, help='int - amount of different clsases in the dataset')
    parser.add_argument('--loss', type=str, default="ce", help='str - how the loss is calculated for the ood sample "bce" ("maxconf" not working yet)')

    parser.add_argument('--epochs', type=int, default=4, help='int - amount of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='float - learning rate of the model')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='float - how much the lr drops after every unsuccessfull step')
    parser.add_argument('--lr_gain', type=float, default=1.1, help='float - how much the lr raises after every successfull step')
    parser.add_argument('--stepsize', type=float, default=0.01, help='float - factor to change the model weights in gradient descent of the adversarial attack')
    parser.add_argument('--num_heads', type=int, default=4, help='int - amount of attention heads for the vit model')
    parser.add_argument('--num_layers', type=int, default=5, help='int - amount of parallel layers doing the calculations for the vit model')
    parser.add_argument('--momentum', type=float, default=0.9, help='float - factor to change the model weights in gradient descent')

    parser.add_argument('--img_size', type=int, default=224, help='int - amount of pixel for the images')
    parser.add_argument('--batch_size', type=int, default=256, help='int - amount of images in the train, valid or test batches')
    parser.add_argument('--workers', type=int, default=0, help='int - amount of workers in the dataloader')

    # boolean parameters, false until the flag is set --> then true
    parser.add_argument('--test', action='store_true', help='flag to set testing to true and test a model')
    parser.add_argument('--save_model', action='store_true', help='flag to save the model if it is being finished training')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # some adjusting of the args, do not remove
    args.batch_size = args.batch_size*2
    args.detector_model_name = None


    #"""
    args.epochs = 2
    args.lr = 0.0001
    args.batch_size = 8 #512  # for training the classifier more than 128 is possible, in the detector it would give a CUDA out of memory --> RuntimeError

    args.save_model = True  # True
    args.test = True  # if True --> Testing, else False --> Training
    args.classification_model_name = "resnet"  # mininet, vit, resnet, cnn_ibp

    args.dataset = "svhn"

    # args.img_size = 112
    #"""

    if args.test:
        test_classifier(args)
    else:
        try:
            train_classifier(args)
        except RuntimeError as error:
            print("Cuda Memory Summary:", torch.cuda.memory_summary(device=None, abbreviated=False))
            print("RuntimeeError:", error)

    print("finished all executions")


