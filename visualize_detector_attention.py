import os
import sys
import argparse
import random
from tqdm import tqdm
import torch
import torchvision
import matplotlib.pyplot as plt

from utils.cifar100_labels import cifar100_labels
from utils.torch_to_pil import tensor_to_pil_image

import vit.src.data_loaders as DataLoader
from vit.src.model import VisionTransformer as ViT
from utils.ood_detection.ood_detector import MiniNet, CNN_IBP
from utils.ood_detection.data_loaders import create_mixed_test_dataloaders
from utils.ood_detection.PGD_attack import MonotonePGD, MaxConf
from utils.store_model import load_classifier, load_detector
from train_detector import get_noise_from_args, shuffle_batch_elements

from utils.cifar10_labels import cifar10_labels
from utils.cifar100_labels import cifar100_labels


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
    elif model_name.lower() == "cnn_ibp":
        #TODO currently not working, throwing error
        #"RuntimeError: mat1 dim 1 must match mat2 dim 0"
        model = CNN_IBP().to(device=args.device)
    elif model_name.lower() == "mininet":
        model = MiniNet().to(device=args.device)
    else:
        raise ValueError("Error - wrong model specified in 'args'")
    return model



def visualize_attn_embeddings(model, img, img_label, ood=False, pert=False, additional_info=""): # img [3, 224, 224]
    # # Detection - Visualize encoder-decoder multi-head attention weights
    # Here we visualize attention weights of the last decoder layer. This corresponds to visualizing, for each
    # detected objects, which part of the image the model was looking at to predict this specific bounding box and class.
    # We will use hooks to extract attention weights (averaged over all heads) from the transformer.

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, cls_token_attn = [], [], []
    hooks = [
        model.transformer.encoder_layers[-2].attn.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ), ]

    # CHANGE not needed here
    # img = transform(img)

    # propagate through the model
    _ = model(img.unsqueeze(0)) # not_contrastive_acc=True

    for hook in hooks:
        hook.remove()

    # enc_attn_weights.append(model.backbone.transformer.blocks[-1].attn.scores)

    # don't need the list anymore
    conv_features = conv_features[0]

    for i in range(len(model.transformer.encoder_layers)):
        cls_token_attn.append(torch.squeeze(model.transformer.encoder_layers[i].attn.score)[:, 0, 1:])
        enc_attn_weights.append(torch.squeeze(model.transformer.encoder_layers[i].attn.score)[:, 1:, 1:])

    im = tensor_to_pil_image(img)

    f_map = (14, 14) # TODO maybe try 16x16
    head_to_show = 2  # change head number based on which attention u want to see

    print_attention_layers = [x for x in range(len(model.transformer.encoder_layers))]
    for i in range(len(model.transformer.encoder_layers)):
        if i in print_attention_layers:  # show only 4th and 11th or last layer
            # get the HxW shape of the feature maps of the ViT
            shape = f_map
            # and reshape the self-attention to a more interpretable shape
            cls_attn = cls_token_attn[i][head_to_show].reshape(shape)
            # print(np.around(cls_attn,2))
            # print('Shape of cls attn : ',cls_attn.shape)
            sattn = enc_attn_weights[i][head_to_show].reshape(shape + shape)
            # print("Reshaped self-attention:", sattn.shape)
            # print('Showing layer {} and and head {}'.format(i,head_to_show))

            fact = 16  # as image size was 160 and number of attn block 160/16=10

            # let's select 3 reference points AND CLASIFICATION TOKEN for visualization in transformed image space
            idxs = [(0, 0), (48, 48), (96, 96), (144, 144), ]

            # here we create the canvas
            fig = plt.figure(constrained_layout=True, figsize=(25 * 0.9, 12 * 0.9))
            # and we add one plot per reference point
            gs = fig.add_gridspec(2, 4)
            axs = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[0, -1]),
                fig.add_subplot(gs[1, -1]),
            ]

            # CHANGE casting to cpu device to be able to cast to numpy
            cls_attn = cls_attn.to(device="cpu")
            sattn = sattn.to(device="cpu")

            # for each one of the reference points, let's plot the self-attention
            # for that point
            for idx_o, ax in zip(idxs, axs):
                if idx_o == (0, 0):
                    idx = (idx_o[0] // fact, idx_o[1] // fact)
                    ax.imshow(cls_attn, cmap='cividis', interpolation='nearest')
                    ax.axis('off')
                    # ax.set_title(f'cls attn at layer: {i}')
                    ax.set_title('cls token attention', fontsize=22)
                else:
                    idx = (idx_o[0] // fact, idx_o[1] // fact)
                    ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
                    ax.axis('off')
                    # ax.set_title(f'self-attn{idx_o} at layer: {i}')
                    ax.set_title(f'self-attention at{idx_o}', fontsize=22)

            # and now let's add the central image, with the reference points as red circles
            fcenter_ax = fig.add_subplot(gs[:, 1:-1])
            # CHANGE removed resize_transform
            #fcenter_ax.imshow(resize_transform(im))  # cls_attn
            fcenter_ax.imshow(im)  # cls_attn
            for (y, x) in idxs:
                if not (x == 0 and y == 0):
                    x = ((x // fact) + 0.5) * fact
                    y = ((y // fact) + 0.5) * fact
                    fcenter_ax.add_patch(plt.Circle((x, y), fact // 3, color='r'))
                    fcenter_ax.axis('off')
            if ood:
                if pert:
                    plt.savefig(f'figures/attention/perturbed/ood_pert_{img_label}{additional_info}_att-layer{i}.png')
                else:
                    plt.savefig(f'figures/attention/clean/ood_clean_{img_label}{additional_info}_att-layer{i}.png')
            else:
                if pert:
                    plt.savefig(f'figures/attention/perturbed/id_pert_{img_label}{additional_info}_att-layer{i}.png')
                else:
                    plt.savefig(f'figures/attention/clean/id_clean_{img_label}{additional_info}_att-layer{i}.png')
            plt.close('all')


def visualize_detector_attention(args):
    classifier = load_classifier(args)
    detector = load_detector(args)
    classifier, detector = classifier.to(args.device), detector.to(args.device)

    # writes the datadirs directly into the args
    set_id_ood_datadirs(args)

    # get an ID and OOD dataloader to visualize one image per dataloader
    if args.dataset == 'svhn': args.dataset = 'SVHN'
    if args.ood_dataset == 'svhn': args.ood_dataset = 'SVHN'

    id_dataloader = eval("DataLoader.{}DataLoader".format(args.dataset))(
        data_dir=args.data_dir,  # os.path.join(config.data_dir, config.dataset),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split='val',
        net=args.model)

    ood_dataloader = eval("DataLoader.{}DataLoader".format(args.ood_dataset))(
        data_dir=args.ood_data_dir,  # os.path.join(config.data_dir, config.dataset),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split='val',
        net=args.model)

    if args.dataset == 'SVHN': args.dataset = 'svhn'
    if args.ood_dataset == 'SVHN': args.ood_dataset = 'svhn'

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

        dataloader_iterator = iter(id_dataloader)

        for index, [ood_inputs, ood_labels] in enumerate(tqdm(ood_dataloader)):
            try:
                [id_inputs, id_labels] = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(id_dataloader)
                [ood_inputs, ood_labels] = next(dataloader_iterator)

            id_inputs, id_labels = id_inputs.to(device=args.device), id_labels.to(device=args.device)
            ood_inputs, ood_labels = ood_inputs.to(device=args.device), ood_labels.to(device=args.device)

            random_selection = random.randint(0, args.batch_size-1)
            id_input, ood_input = id_inputs[random_selection], ood_inputs[random_selection]
            id_label, ood_label = id_labels[random_selection], ood_labels[random_selection]

            id_label_string = get_string_label_for_sample(id_label, args.dataset)
            ood_label_string = get_string_label_for_sample(ood_label, args.ood_dataset)

            visualize_attn_embeddings(detector, id_input, id_label_string, ood=False, pert=False)
            visualize_attn_embeddings(detector, ood_input, ood_label_string, ood=True, pert=False)

            if attack:
                perturbed_id_inputs, _, _ = attack(id_inputs, id_labels)
                perturbed_ood_inputs, _, _ = attack(ood_inputs, ood_labels)
                id_input, ood_input = id_inputs[random_selection], ood_inputs[random_selection]

                visualize_attn_embeddings(detector, id_input, id_label_string, ood=False, pert=True)
                visualize_attn_embeddings(detector, ood_input, ood_label_string, ood=True, pert=True)

            if index == args.visualize: break

    print("Finished visualizing the detector attention")


def set_id_ood_datadirs(args):
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


def get_string_label_for_sample(label, dataset):
    if dataset == "cifar10":
        return cifar10_labels[label]
    elif dataset == "cifar100":
        return cifar100_labels[label]
    elif dataset == "svhn":
        return str(label.item())
    else:
        raise ValueError("dataset not supported, labels could not get fetched")


def parse_args():
    parser = argparse.ArgumentParser(description='Run the monotone PGD attack on a batch of images, default is with ViT and the MPGD of Alex, where cifar10 is ID and cifar100 is OOD')

    parser.add_argument('--model', type=str, default="vit", help='str - what model should be used to classify input samples "vit", "resnet", "mininet" or "cnn_ibp"')
    parser.add_argument("--classification-ckpt", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--detector-ckpt-path", type=str, default=None, help="model checkpoint to load weights")

    parser.add_argument('--device', type=str, default="cuda", help='str - cpu or cuda to calculate the tensors on')
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--select-gpu", type=int, default=0, help="select gpu to use, no parallelization possible")
    parser.add_argument('--attack', action='store_true', help='toggels the MonotonePGD attack')
    parser.add_argument("--visualize", type=int, default=3, help="amount of id and ood images to visualize")

    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')
    parser.add_argument('--contrastive', action='store_true', help='using distributed loss calculations across multiple GPUs')

    return parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
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

    visualize_detector_attention(args)
    print("finished all executions")

