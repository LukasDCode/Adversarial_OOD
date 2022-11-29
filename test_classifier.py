import argparse
from tqdm import tqdm
import torch
import torchvision
import numpy as np

from vit.src.model import VisionTransformer as ViT
from vit.src.utils import MetricTracker, accuracy
from utils.ood_detection.ood_detector import MiniNet, CNN_IBP
from utils.normalize_image_data import normalize_general_image_data, normalize_cifar10_image_data, normalize_cifar100_image_data, normalize_SVHN_image_data
#from train_ood_detector import get_model_from_args
from utils.load_data import get_train_valid_dataloaders, get_test_dataloader
from utils.store_model import save_model, load_model


def get_model_from_args(args, model_name, num_classes):
    if model_name.lower() == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device=args.device) # cuda()
    elif model_name.lower() == "vit":
        model = ViT(image_size=(args.img_size, args.img_size),  # 224,224
                    num_heads=args.num_heads, #12 #also a very small amount of heads to speed up training
                    num_layers=args.num_layers,  # 12 # 5 is a very small vit model
                    num_classes=num_classes,  # 2 for OOD detection, 10 or more for classification
                    contrastive=False,
                    timm=True).to(device=args.device) # cuda()
        """
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        inputs = feature_extractor(image, return_tensors="pt")
        """
    elif model_name.lower() == "cnn_ibp":
        #TODO currently not working, throwing error
        #"RuntimeError: mat1 dim 1 must match mat2 dim 0"
        model = CNN_IBP().to(device=args.device)
    elif model_name.lower() == "mininet":
        model = MiniNet().to(device=args.device)
    else:
        raise ValueError("Error - wrong model specified in 'args'")
    return model


def train_classifier(args):
    if args.device == "cuda": torch.cuda.empty_cache()

    classification_model = get_model_from_args(args, args.model, num_classes=args.num_classes)
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
        save_model(args, classification_model, optimizer)


def train_one_epoch(args, classification_model, loss_fn, optimizer, train_dataloader):
    running_loss, last_loss = 0., 0.

    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        # some dataloaders return a list within a list with duplicates, but we only need one of those doubles
        if isinstance(inputs, list):
            inputs = inputs[0]

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

        print_interval = 16384/args.batch_size
        if i % print_interval == print_interval - 1:
            last_loss = running_loss / print_interval  # loss per batch
            print("   Batch", i + 1, "loss:", last_loss)
            running_loss = 0.

    return last_loss


def test_classifier(args):
    model = load_model(args)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    metrics = MetricTracker(*[metric for metric in metric_names], writer=None)
    log = {}

    # get a dataloader mixed 50:50 with ID and OOD data and labels of 0 (ID) and 1 (OOD)
    test_dataloader = get_test_dataloader(args)

    print("Start Testing")
    with torch.no_grad():
        model.eval()
        running_test_error = 0
        for epoch_nr, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            # from ViT training validation
            metrics.reset()

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

            # from ViT training validation
            # valid_dataloader = cifar10-Dataloader-Object
            # criterion = CrossEntropyLoss()
            # valid_metrics = MetricTracker
            # device = device cuda or cpu
            #result = valid_epoch(epoch_nr+1, model, valid_dataloader, criterion, valid_metrics, device)

            losses = []
            acc1s = []
            acc5s = []
            criterion = torch.nn.CrossEntropyLoss()

            loss = criterion(outputs, labels)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

        loss = np.mean(losses)
        acc1 = np.mean(acc1s)
        acc5 = np.mean(acc5s)
        if metrics.writer is not None:
            metrics.writer.set_step(epoch_nr, 'valid')
        metrics.update('loss', loss)
        metrics.update('acc1', acc1)
        metrics.update('acc5', acc5)

        log.update(**{'val_' + k: v for k, v in metrics.result().items()})

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))

        # Error
        avg_valid_error = running_test_error / (epoch_nr  + 1)
        print("Old metrics")
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

    parser.add_argument('--model', type=str, default="vit", help='str - what model should be used to classify input samples "vit", "resnet", "mininet" or "cnn_ibp"')
    parser.add_argument('--classification_ckpt', type=str, default=None, help='str - path of pretrained model checkpoint')
    parser.add_argument('--device', type=str, default="cuda", help='str - cpu or cuda to calculate the tensors on')
    #parser.add_argument('--dataset', type=str, default="cifar10", help='str - the in-distribution dataset "cifar10", "cifar100" or "svhn"')
    #parser.add_argument('--num_classes', type=int, default=10, help='int - amount of different clsases in the dataset')
    #parser.add_argument('--loss', type=str, default="ce", help='str - how the loss is calculated for the ood sample "bce" ("maxconf" not working yet)')

    #parser.add_argument('--epochs', type=int, default=10, help='int - amount of training epochs')
    #parser.add_argument('--lr', type=float, default=0.01, help='float - learning rate of the model')
    #parser.add_argument('--lr_decay', type=float, default=0.5, help='float - how much the lr drops after every unsuccessfull step')
    #parser.add_argument('--lr_gain', type=float, default=1.1, help='float - how much the lr raises after every successfull step')
    #parser.add_argument('--momentum', type=float, default=0.9, help='float - factor to change the model weights in gradient descent')
    #parser.add_argument('--stepsize', type=float, default=0.01, help='float - factor to change the model weights in gradient descent of the adversarial attack')
    #parser.add_argument('--num_heads', type=int, default=12, help='int - amount of attention heads for the vit model')
    #parser.add_argument('--num_layers', type=int, default=12, help='int - amount of parallel layers doing the calculations for the vit model')

    #parser.add_argument('--img_size', type=int, default=32, help='int - amount of pixel for the images')
    #parser.add_argument('--batch_size', type=int, default=256, help='int - amount of images in the train, valid or test batches')
    #parser.add_argument('--workers', type=int, default=0, help='int - amount of workers in the dataloader')

    # boolean parameters, false until the flag is set --> then true
    #parser.add_argument('--test', action='store_true', help='flag to set testing to true and test a model')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # some adjusting of the args, do not remove
    #args.batch_size = args.batch_size*2
    #args.detector_model_name = None

    # """
    #args.epochs = 2
    #args.lr = 0.001
    #args.batch_size = 512 #512  # for training the classifier more than 128 is possible, in the detector it would give a CUDA out of memory --> RuntimeError
    #args.patch_size = 16
    #args.emb_dim = 768
    #args.mlp_dim = 3072

    # args.attn_dropout_rate = 0.0
    # args.dropout_rate = 0.1
    # args.head = None
    # args.dataset = "cifar100"
    # args.num_classes = 10

    #args.save_model = False  # True

    """
    args.test = True  # if True --> Testing, else False --> Training
    args.model = "vit"  # mininet, vit, resnet, cnn_ibp
    args.classification_ckpt = "saved_models/trained_classifier/vit_b16_224SupCE_cifar100_bs64_best_accuracy.pth"
    """

    if args.test:
        test_classifier(args)
    else:
        try:
            train_classifier(args)
        except RuntimeError as error:
            print("Cuda Memory Summary:", torch.cuda.memory_summary(device=None, abbreviated=False))
            print("RuntimeeError:", error)

    print("finished all executions")

