import os
import sys
import argparse
from tqdm import tqdm
import torch
import torchvision
import numpy as np
import sklearn.metrics

from vit.src.model import VisionTransformer as ViT
from vit.src.utils import MetricTracker, accuracy
from utils.ood_detection.ood_detector import MiniNet, CNN_IBP
from utils.ood_detection.data_loaders import create_mixed_test_dataloaders
from utils.ood_detection.PGD_attack import MonotonePGD, MaxConf
from utils.store_model import load_classifier, load_detector
from train_detector import get_noise_from_args, shuffle_batch_elements


def get_model_from_args(args, model_name, num_classes):
    """
    get_model_from_args loads a classifier model from the specified arguments dotdict

    :args: dotdict containing all the arguments
    :model_name: string stating what kind of model should be loaded as a classifier
    :num_classes: integer specifying how many classes the classifier should be able to detect
    :return: loaded classifier model
    """
    if model_name.lower() == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device=args.device) # cuda()
    elif model_name.lower() == "vit":
        model = ViT(image_size=(args.img_size, args.img_size),  # 224,224
                    num_heads=args.num_heads, #12 #also a very small amount of heads to speed up training
                    num_layers=args.num_layers,  # 12 # 5 is a very small vit model
                    num_classes=num_classes,  # 2 for OOD detection, 10 or more for classification
                    contrastive=False,
                    timm=True).to(device=args.device) # cuda()
    elif model_name.lower() == "cnn_ibp":
        #TODO currently not working, throwing error
        #"RuntimeError: mat1 dim 1 must match mat2 dim 0"
        model = CNN_IBP().to(device=args.device)
    elif model_name.lower() == "mininet":
        model = MiniNet().to(device=args.device)
    else:
        raise ValueError("Error - wrong model specified in 'args'")
    return model


def test_detector(args):
    """
    test_classifier runs one epoch of all test samples to evaluate the detectors' performance.

    :args: dotdict containing all the arguments
    """
    classifier = load_classifier(args)
    detector = load_detector(args)
    classifier, detector = classifier.to(args.device), detector.to(args.device)

    # CHANGE # TODO remove
    args.batch_size = 16

    # writes the datadirs directly into the args
    set_id_ood_datadirs(args)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc2']
    metrics = MetricTracker(*[metric for metric in metric_names], writer=None)
    log = {}
    losses = []
    acc1s = []
    acc2s = []
    auroc_list, aupr_list = [], [] # lists for the auroc and aupr values of all samples

    # get a dataloader mixed 50:50 with ID and OOD data and labels of 0 (ID) and 1 (OOD)
    if args.dataset == 'svhn': args.dataset = 'SVHN'
    test_dataloader = create_mixed_test_dataloaders(args)
    if args.dataset == 'SVHN': args.dataset = 'svhn'

    # attack instance
    if args.attack:
        noise = get_noise_from_args(args.noise, args.eps)
        attack = MonotonePGD(args.eps, args.iterations, args.stepsize, num_classes=2, momentum=0.9,
                             norm=args.norm, loss=MaxConf(True), normalize_grad=False, early_stopping=0,
                             restarts=args.restarts, init_noise_generator=noise, model=classifier, save_trajectory=False)
    else:
        attack = None


    with torch.no_grad():
        classifier.eval()
        detector.eval()
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        for batch_nr, (id_data, ood_data) in enumerate(tqdm(test_dataloader)):
            if args.device == "cuda": torch.cuda.empty_cache()
            inputs, labels = shuffle_batch_elements(id_data, ood_data)
            # from ViT training validation
            metrics.reset()
            inputs, labels = inputs.to(device=args.device), labels.to(device=args.device)

            outputs = detector(inputs, not_contrastive_acc=True)
            #if args.device == "cuda": torch.cuda.empty_cache()

            loss = criterion(outputs, labels)
            acc1, acc2 = accuracy(outputs, labels, topk=(1, 2))
            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc2s.append(acc2.item())

            aupr_list.append(sklearn.metrics.average_precision_score(labels.to(device="cpu"), outputs[:, 1].to(device="cpu")))
            auroc_list.append(sklearn.metrics.roc_auc_score(labels.to(device="cpu"), outputs[:, 1].to(device="cpu")))

            if attack:
                perturbed_inputs, _, _ = attack(inputs, labels)
                p_outputs = detector(perturbed_inputs, not_contrastive_acc=True)

                loss = criterion(p_outputs, labels)
                acc1, acc2 = accuracy(p_outputs, labels, topk=(1, 2))
                losses.append(loss.item())
                acc1s.append(acc1.item())
                acc2s.append(acc2.item())

                aupr_list.append(sklearn.metrics.average_precision_score(labels.to(device="cpu"), p_outputs[:, 1].to(device="cpu")))
                auroc_list.append(sklearn.metrics.roc_auc_score(labels.to(device="cpu"), p_outputs[:, 1].to(device="cpu")))

            if args.device == "cuda": torch.cuda.empty_cache()

            # break out of loop sooner, because a testing takes around 16h equal to one epoch of training, 1 iteration takes ~20sec
            if args.break_early and batch_nr == 1200: break

    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc2 = np.mean(acc2s)
    if metrics.writer is not None:
        metrics.writer.set_step(batch_nr, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('acc2', acc2)

    log.update(**{'val_' + k: v for k, v in metrics.result().items()})

    # print logged informations to the screen
    for key, value in log.items():
        print('    {:15s}: {}'.format(str(key), value))

    print("ID:", args.dataset, " ---  OOD:", args.ood_dataset)
    print("AUROC: ", sum(auroc_list)/len(auroc_list))
    print("AUPR:  ", sum(aupr_list)/len(aupr_list))

    print("Finished Testing the Model")


def set_id_ood_datadirs(args):
    """
    set_id_ood_datadirs modifies the arguments dotdict to contain the right paths where to load the datasets from.

    :args: dotdict containing all arguments (including the paths of the id and ood datasets)
    """
    if args.dataset.lower() == "cifar10":
        args.data_dir = "data/cifar10/"
    elif args.dataset.lower() == "cifar100":
        args.data_dir = "data/cifar100/"
    elif args.dataset.lower() == "svhn":
        args.data_dir = "data/"
    else:
        raise ValueError("Unknown ID dataset specified, no path to the dataset available")
    if args.ood_dataset.lower() == "cifar10":
        args.ood_data_dir = "data/cifar10/"
    elif args.ood_dataset.lower() == "cifar100":
        args.ood_data_dir = "data/cifar100/"
    elif args.ood_dataset.lower() == "svhn":
        args.ood_data_dir = "data/"
    else:
        raise ValueError("Unknown OOD dataset specified, no path to the dataset available")


def parse_args():
    """
    parse_args retrieves the arguments from the command line and parses them into the arguments dotdict.

    :return: dotdict with all the arguments
    """
    parser = argparse.ArgumentParser(description='Run the monotone PGD attack on a batch of images, default is with ViT and the MPGD of Alex, where cifar10 is ID and cifar100 is OOD')

    parser.add_argument('--model', type=str, default="vit", help='str - what model should be used to classify input samples "vit", "resnet", "mininet" or "cnn_ibp"')
    parser.add_argument("--classification-ckpt", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--detector-ckpt-path", type=str, default=None, help="model checkpoint to load weights")

    parser.add_argument('--device', type=str, default="cuda", help='str - cpu or cuda to calculate the tensors on')
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--select-gpu", type=int, default=0, help="select gpu to use, no parallelization possible")
    parser.add_argument('--attack', action='store_true', help='toggels the MonotonePGD attack')
    parser.add_argument('--break-early', action='store_true', help='interupt execution earlier for developing purposes')

    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')
    parser.add_argument('--contrastive', action='store_true', help='using distributed loss calculations across multiple GPUs')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device_id_list = list(range(torch.cuda.device_count()))
    # it is NOT possible to Parallelize the attack, so cuda is set to one GPU and device_ids are set to None
    if args.device == "cuda":
        if args.select_gpu in device_id_list:
            # allows to specify a certain cuda core, because parallelization is not possible
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.select_gpu)
        else:
            print("Specified selected GPU is not available, not that many GPUs available --> Defaulted to cuda:0")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print("num_workers is set to 0 in debugging mode, otherwise debugging issues occur")
        args.num_workers = 0

    test_detector(args)
    print("finished all executions\n")

