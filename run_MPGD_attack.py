#!/usr/bin/env python

import argparse
import torch
torch.set_grad_enabled(False)
import torchvision
from vit.src.model import VisionTransformer as ViT
from PGD_Alex import UniformNoiseGenerator, NormalNoiseGenerator, Contraster, DeContraster
from PGD_Alex import MonotonePGD, MaxConf, MonotonePGD_Lukas
from vit.src.data_loaders import create_dataloaders

from perturb_images import perturb_image_batch, perturb_single_image
from utils.dotdict import dotdict


def parse_args():
    parser = argparse.ArgumentParser(description='Run the monotone PGD attack on a batch of images, default is with ViT and the MPGD of Alex, where cifar10 is ID and cifar100 is OOD')

    parser.add_argument('--model', type=str, default="vit", help='str - what model should be used for the attack')
    parser.add_argument('--ckpt', type=str, default="/nfs/data3/koner/contrastive_ood/save/vit/vit_224SupCE_cifar10_bs512_lr0.01_wd1e-05_temp_0.1_210316_122535/checkpoints/ckpt_epoch_50.pth", help='str - path of pretrained model checkpoint')
    parser.add_argument('--device', type=str, default="cpu", help='str - cpu or cuda to calculate the tensors on')
    parser.add_argument('--in_dataset', type=str, default="cifar10", help='str - the in-distribution dataset (currently only cifar10)')
    parser.add_argument('--out_dataset', type=str, default="cifar100", help='str - the out-distribution dataset (currently only cifar100)')
    parser.add_argument('--loss', type=str, default="maxconf", help='str - how the loss is calculated for the ood sample (currently only maxconf)')

    parser.add_argument('--eps', type=float, default=0.01, help='float - the radius of the max perturbation ball around the sample')
    parser.add_argument('--norm', type=str, default="inf", help='str - inf or l2 norm (currently only inf)')
    parser.add_argument('--iterations', type=int, default=15, help='int - how many steps of perturbations for each restart')
    parser.add_argument('--restarts', type=int, default=2, help='int - how often the MPGD attack starts over at a random place in its eps-space')
    parser.add_argument('--noise', type=str, default="normal", help='str - normal, uniform, contraster or decontraster noise is possible')

    parser.add_argument('--lr_decay', type=float, default=0.5, help='float - how much the lr drops after every unsuccessfull step')
    parser.add_argument('--lr_gain', type=float, default=1.1, help='float - how much the lr raises after every successfull step')
    parser.add_argument('--stepsize', type=float, default=0.01, help='float - factor to change the model weights in gradient descent')

    parser.add_argument('--imagesize', type=int, default=224, help='int - amount of pixel for the images')
    parser.add_argument('--batchsize', type=int, default=16, help='int - amount of images in one of the train or valid batches')
    parser.add_argument('--workers', type=int, default=0, help='int - amount of workers in the dataloader')

    # boolean parameters, false until the flag is set --> then true
    parser.add_argument('--visualize', action='store_true', help='store the original & perturbed images and the attention maps as png files')
    parser.add_argument('--single', action='store_true', help='only perform the attack for a single image of the batch, no parallelization')
    parser.add_argument('--lukas', action='store_true', help='use the homemade mpgd attack, not the one from alex (currently under maintenance)')

    return parser.parse_args()


def perform_MPGD_attack(args):
    # HYPERPARAMETERS
    IMAGE_SIZE = args.imagesize #224

    if args.in_dataset == "cifar10":
        num_classes = 10
        #classes = cifar10_labels  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        print("The in_dataset is not correctly specified. Recommended to use the default (cifar10).")
        raise NotImplementedError


    if args.model == "vit":
        num_heads = 12
        num_layers = 12
        # load ViT model and its ckpt
        ckpt = torch.load(args.ckpt, map_location=torch.device(args.device))
        model = ViT(image_size=(IMAGE_SIZE,IMAGE_SIZE), #224,224
                     num_heads=num_heads, #12
                     num_layers=num_layers, #12
                     num_classes=num_classes, #10
                     contrastive=False,
                     timm=True)
        # now load the model
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model = model.to(device=args.device)#cuda()
        model.eval()
        print('ViT Model loaded....')
    else:
        model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        print("ResNet Model loaded....")


    dataloaders_config = {
        "data_dir": "/home/wiss/koner/Lukas/adversarial_ood/data/cifar-100-python/", # "/home/koner/adversarial_ood/data/cifar-100-python/"
        "image_size": IMAGE_SIZE, #224
        "batch_size": args.batchsize, #16
        "num_workers": args.workers, #0
        "contrastive": False,
        "albumentation": False,
        "net": "vit", # does not even get used inside create_dataloader function
        "no_train_aug": False,
        "dataset": args.out_dataset, #"cifar100"
        "deit": False
    }
    dataloaders_config = dotdict(dataloaders_config)
    train_dataloader, valid_dataloader = create_dataloaders(dataloaders_config)
    # here both dataloaders have values of [0;255], the values change later on


    if args.loss == "maxconf":
        loss = MaxConf(True)
    else:
        print("The loss is not correctly specified. Recommended to use the default (maxconf).")
        raise NotImplementedError


    if args.noise.lower() == "normal":
        noise = NormalNoiseGenerator(sigma=1e-4)
    elif args.noise.lower() == "uniform":
        noise = UniformNoiseGenerator(min=-args.eps, max=args.eps)
    elif args.noise.lower() == "contraster":
        noise = Contraster(args.eps)
    elif args.noise.lower() == "decontraster":
        noise = DeContraster(args.eps)
    else:
        noise = NormalNoiseGenerator(sigma=1e-4)


    if args.lukas:
        print("+++++ LUKAS MonotonePGD +++++")
        attack = MonotonePGD_Lukas(args.eps, args.iterations, args.stepsize, num_classes, momentum=0.9, norm=args.norm,
                                   loss=loss, normalize_grad=False, early_stopping=0, restarts=args.restarts,
                                   init_noise_generator=noise, model=model, save_trajectory=False)
    else:
        print("+++++ ALEXANDER MonotonePGD +++++")
        attack = MonotonePGD(args.eps, args.iterations, args.stepsize, num_classes, momentum=0.9, norm=args.norm,
                             loss=loss, normalize_grad=False, early_stopping=0, restarts=args.restarts,
                             init_noise_generator=noise, model=model, save_trajectory=False)


    if args.single:
        clean_out_list, perturbed_out_list = perturb_single_image(model, attack, train_dataloader, args.device,
                                                                  visualization=args.visualize)
    else:
        clean_out_list, perturbed_out_list = perturb_image_batch(model, attack, train_dataloader, args.device,
                                                                 visualization=args.visualize)


if __name__ == "__main__":
    args = parse_args()
    perform_MPGD_attack(args)

    # this can be deleted, it is only for a git test

