import os
import sys
import random
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from vit.src.model import VisionTransformer
from vit.src.config import *
from vit.src.checkpoint import load_checkpoint
from vit.src.utils import accuracy, MetricTracker
from utils.ood_detection.data_loaders import create_dataloaders
from utils.get_model import get_model_from_args
from util import adjust_learning_rate
from losses import SupConLoss

from utils.ood_detection.PGD_attack import MonotonePGD, MaxConf, get_noise_from_args

def train_epoch(epoch, detector, data_loader, criterion, optimizer, attack, lr_scheduler, metrics,
                device=torch.device('cpu'), contrastive=False, test_contrastive_acc=False, method=None, criterion2=None, head=None, mixup_fn=None, break_early=False):
    metrics.reset()

    # training loop
    for batch_idx, (id_data, ood_data) in enumerate(tqdm(data_loader)):
        # here the returned batch consists of 50% ID data and 50% OOD data
        # they get shuffeled inside the batch and the labels adjusted to 0 for ID and 1 for OOD
        batch_data, batch_target = shuffle_batch_elements(id_data, ood_data)
        # batch_data = [batch_size, channels=3, image_size=224, image_size=224]
        # batch_target = [batch_size]
        if contrastive:
            if isinstance(batch_data[0],dict):#for albumnetations
                batch_data = torch.cat([batch_data[0]['image'], batch_data[1]['image']], dim=0)  # .to(device)
            else:
                batch_data = torch.cat([batch_data[0], batch_data[1]], dim=0)#.to(device)
        else:
            batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        if mixup_fn is not None and not contrastive:
            batch_data, batch_target = mixup_fn(batch_data,batch_target)

        optimizer.zero_grad()
        if head == "both" and contrastive:
            batch_pred, pred_classifier = detector(batch_data, not_contrastive_acc = not test_contrastive_acc)
        else:
            # batch_data values are in range [0;1]
            batch_pred = detector(batch_data, not_contrastive_acc=not test_contrastive_acc)


        if contrastive:
            bsz = batch_target.shape[0]
            f1, f2 = torch.split(batch_pred, [bsz, bsz], dim=0)
            batch_pred = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if method == 'SimCLR':
            loss = criterion(batch_pred)
        else:
            loss = criterion(batch_pred, batch_target)

        if head=="both" and contrastive:# train both head contrastive and classifier
            loss1 = criterion2(pred_classifier, batch_target)
            loss = loss + 0.2*loss1
        loss.backward()
        optimizer.step()


        if attack:
            # Here also the ID data gets perturbed, but actually it is just an augmentation
            # calls the perturb() of RestartAttack --> which calls perturb_inner() of MonotonePGD
            perturbed_inputs, _, _ = attack(batch_data, batch_target)  # once WAS inputs.clone()
            optimizer.zero_grad()  # Zero gradients for every batch

            if head == "both" and contrastive:
                perturbed_batch_pred, pred_classifier = detector(perturbed_inputs, not_contrastive_acc=not test_contrastive_acc)
            else:
                # batch_data values are in range [0;1]
                perturbed_batch_pred = detector(perturbed_inputs, not_contrastive_acc=not test_contrastive_acc)

            if contrastive:
                bsz = batch_target.shape[0]
                f1, f2 = torch.split(perturbed_batch_pred, [bsz, bsz], dim=0)
                perturbed_batch_pred = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if method == 'SimCLR':
                loss = criterion(perturbed_batch_pred)
            else:
                loss = criterion(perturbed_batch_pred, batch_target)

            if head == "both" and contrastive:  # train both head contrastive and classifier
                loss1 = criterion2(pred_classifier, batch_target)
                loss = loss + 0.2 * loss1

            loss.backward()
            optimizer.step()  # Adjust learning weights


        if lr_scheduler is not None:
            lr_scheduler.step()
        #torch.cuda.empty_cache()
        metrics.update('loss', loss.item())
        if mixup_fn is None: # for mixup dont calculate accuracy
            if batch_idx % 100 == 10 and not contrastive:
                acc1, acc2 = accuracy(batch_pred, batch_target, topk=(1, 2)) # (1,2) because when only two classes then no acc5 possible, only acc2
                metrics.update('acc1', acc1.item())
                metrics.update('acc2', acc2.item())

                p_acc1, p_acc2 = accuracy(perturbed_batch_pred, batch_target, topk=(1, 2))  # (1,2) because when only two classes then no acc5 possible, only acc2
                metrics.update('acc1', p_acc1.item())
                metrics.update('acc2', p_acc2.item())

            if batch_idx % 100 == 10 and not test_contrastive_acc:
                print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, Acc@2: {:.2f}"
                        .format(epoch, batch_idx, len(data_loader), loss.item(), 0 if contrastive else acc1.item(),0 if contrastive else acc2.item()))#, acc5.item()

        # break out of loop sooner, because a full epoch takes around 16h, 1 iteration takes ~20sec
        if break_early and batch_idx == 10: break

    return metrics.result()


def valid_epoch(epoch, detector, attack, data_loader, criterion, metrics, device=torch.device('cpu'), break_early=False):
    metrics.reset()
    losses = []
    acc1s = []
    acc2s = []
    criterion = torch.nn.CrossEntropyLoss()
    # validation loop
    with torch.no_grad():
        # for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader)):
        for batch_idx, (id_data, ood_data) in enumerate(tqdm(data_loader)):
            # here the returned batch consists of 50% ID data and 50% OOD data
            # they get shuffeled inside the batch and the labels adjusted to 0 for ID and 1 for OOD
            batch_data, batch_target = shuffle_batch_elements(id_data, ood_data)
            # batch_data = [batch_size, channels=3, image_size=224, image_size=224]
            # batch_target = [batch_size]
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = detector(batch_data, eval = args.eval)
            loss = criterion(batch_pred, batch_target)
            acc1, acc2 = accuracy(batch_pred, batch_target, topk=(1, 2)) # (1,2) because when only two classes then no acc5 possible, only acc2
            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc2s.append(acc2.item())


            if attack:
                # Here also the ID data gets perturbed, but actually it is just an augmentation
                # calls the perturb() of RestartAttack --> which calls perturb_inner() of MonotonePGD
                perturbed_inputs, _, _ = attack(batch_data, batch_target)  # once WAS inputs.clone()

                perturbed_batch_pred = detector(perturbed_inputs, eval=args.eval)
                loss = criterion(perturbed_batch_pred, batch_target)
                p_acc1, p_acc2 = accuracy(perturbed_batch_pred, batch_target, topk=(1, 2))  # (1,2) because when only two classes then no acc5 possible, only acc2
                losses.append(loss.item())
                acc1s.append(p_acc1.item())
                acc2s.append(p_acc2.item())

            # break out of validation sooner, because a full validation takes around 16h same as 1 epoch, 1 iteration takes ~20sec
            if break_early and batch_idx == 10: break
            elif batch_idx == 200: break # TODO remove this is for validation to only take 1h instead of 3.5h


    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc2 = np.mean(acc2s)
    if metrics.writer is not None:
        metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('acc2', acc2)
    return metrics.result()


def main(args, device):
    # metric tracker
    metric_names = ['loss', 'acc1', 'acc2']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=None)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=None)

    # create detector model
    print("create detector model")
    detector = VisionTransformer(
             image_size=(args.image_size, args.image_size),
             patch_size=(args.patch_size, args.patch_size),
             emb_dim=args.emb_dim,
             mlp_dim=args.mlp_dim,
             num_heads=args.num_heads,
             num_layers=args.num_layers,
             num_classes=args.num_classes,
             attn_dropout_rate=args.attn_dropout_rate,
             dropout_rate=args.dropout_rate,
             contrastive=args.contrastive,
             timm=True,
             head=args.head)#'jx' in args.checkpoint_path)

    classifier = load_classifier(copy.copy(args))

    # for cutmix and mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Activating cutmix and mixup")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)


    # load checkpoint
    if args.checkpoint_path:
        state_dict = load_checkpoint(args.checkpoint_path, new_img=args.image_size, emb_dim=args.emb_dim,
                                     layers=args.num_layers,patch=args.patch_size)
        print("Loading pretrained weights from {}".format(args.checkpoint_path))
        if not args.test_contrastive_acc and  not args.eval and args.num_classes != state_dict['classifier.weight'].size(0)  :#not
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print("re-initialize fc layer")
            missing_keys = detector.load_state_dict(state_dict, strict=False)
        else:
            missing_keys = detector.load_state_dict(state_dict, strict=False)
        print("Missing keys from checkpoint ",missing_keys.missing_keys)
        print("Unexpected keys in network : ",missing_keys.unexpected_keys)


    # send detector and classifier model to device
    detector = detector.to(device)
    classifier = classifier.to(device)

    # create dataloader
    # CHANGE svhn dataloader is in capital letters, later reset to lower case
    if args.dataset == 'svhn': args.dataset = 'SVHN'
    train_dataloader, valid_dataloader = create_dataloaders(args)
    if args.dataset == 'SVHN': args.dataset = 'svhn'
    # training criterion
    print("create criterion and optimizer")
    if args.contrastive:
        print("Using contrastive loss...")
        criterion = SupConLoss(temperature=args.temp, similarity_metric=args.sim_metric).to(device)
    else:
        if args.mixup > 0.:
            print("Criterion using mixup ")
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy().to(device)
        elif args.smoothing:
            print("Criterion using labelsmoothong ")
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        else:
            print("Criterion using only crossentropy ")
            criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.contrastive and args.head=="both":
        print("Using both loss of supcon and crossentropy")
        criterion2 = nn.CrossEntropyLoss().to(device)
    else:
        criterion2 = None


    # create optimizers and learning rate scheduler
    if args.opt =="AdamW":
        print("Using AdmW optimizer")
        optimizer = torch.optim.AdamW(params=detector.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(
            params=detector.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            momentum=0.9)
    if args.cosine:
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr,
            pct_start=args.warmup_steps / args.train_steps,
            total_steps=args.train_steps)


    # attack instance
    if args.attack:
        noise = get_noise_from_args(args.noise, args.eps)
        attack = MonotonePGD(args.eps, args.iterations, args.stepsize, num_classes=2, momentum=0.9, norm=args.norm,
                             loss=MaxConf(True), normalize_grad=False, early_stopping=0, restarts=args.restarts,
                             init_noise_generator=noise, model=classifier,
                             save_trajectory=False)
    else:
        attack = None


    # start training
    print("start training")
    best_acc = 0.0
    best_epoch = 0
    args.epochs = args.train_steps // len(train_dataloader)
    print("length of train loader : ", len(train_dataloader), ' and total epoch ', args.epochs)
    for epoch in range(1, args.epochs + 1):
        if args.cosine:
            adjust_learning_rate(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            print("learning rate at {0} epoch is {1}".format(epoch, param_group['lr']))

        log = {'epoch': epoch}

        if not args.eval:
            # train the detector model
            detector.train()
            result = train_epoch(epoch, detector, train_dataloader, criterion, optimizer, attack, lr_scheduler, train_metrics, device,
                                 contrastive=args.contrastive, test_contrastive_acc=args.test_contrastive_acc, method=args.method,
                                 head=args.head, criterion2=criterion2, mixup_fn=mixup_fn, break_early=args.break_early)
            log.update(result)

        # validate the detector model
        if not args.contrastive:
            detector.eval()
            result = valid_epoch(epoch, detector, attack, valid_dataloader, criterion, valid_metrics, device, break_early=args.break_early)
            log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        if log['val_acc1'] > best_acc:
            best_acc = log['val_acc1']
            best_epoch = epoch
            best = True

            # CHANGE
            # save the detector model with the best accuracy
            print("save the detector model with the best accuracy")
            save_vit_detector(args, detector, optimizer, epoch)


        # print logged information to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))

        # a full epoch takes ~16h to execute and n epochs take n times as much time
        if args.break_early and epoch == 1: break


    if args.test_contrastive_acc or args.eval or not args.contrastive:
        print("Best accuracy : ", best_acc, ' for ', best_epoch)# saving class mean
        #best_curr_acc = {'best_acc': best_acc,'best_epoch': best_epoch,
        #                 'curr_acc': log['val_acc1'],'curr_epoch': epoch}


def shuffle_batch_elements(data_id, data_ood):
    """
    shuffles the samples of ID and OOD data batch --> return: shuffled_inputs
    also shuffles the labels of ID and OOD of the same data batch in the same order --> return: shuffled_targets
    in the end every shuffled_inputs shuffled_targets tensor contains batch_size/2 samples of the ID and
    batch_size/2 samples of the OOD dataset (default is 32 ID + 32 OOD)
    """
    randomness = list(range(data_id[1].size(dim=0) + data_ood[1].size(dim=0))) # list of indices will determine the shuffle of inputs & targets alike
    random.shuffle(randomness)  # shuffle tensor elements randomly

    shuffled_inputs = torch.cat((data_id[0], data_ood[0]), 0)[randomness]
    shuffled_targets = torch.cat((torch.zeros(data_id[1].size(dim=0)), torch.ones(data_ood[1].size(dim=0))), 0).type(torch.int64)[randomness]

    del data_id[1], data_ood[1] # remove the unused original class labels for performance
    return shuffled_inputs, shuffled_targets



def load_classifier(args):
    if args.classifier_ckpt_path:  # args.classification_ckpt #checkpoint_path
        checkpoint_path = args.classifier_ckpt_path  # args.classification_ckpt
    else:
        raise ValueError("No checkpoint path for the classifier was specified")

    checkpoint = torch.load(checkpoint_path)

    args.dataset = checkpoint['dataset']

    args.model = checkpoint['model_name']
    args.method = checkpoint['loss']
    args.num_classes = checkpoint['num_classes'] # 10 or 100

    args.image_size = checkpoint['image_size']
    args.batch_size = checkpoint['batch_size'] # 64
    args.patch_size = checkpoint['patch_size'] # 16

    args.emb_dim = checkpoint['emb_dim'] # 768
    args.mlp_dim = checkpoint['mlp_dim'] # 3072
    args.num_heads = checkpoint['num_heads'] # 12
    args.num_layers = checkpoint['num_layers'] # 12
    args.attn_dropout_rate = checkpoint['attn_dropout_rate']
    args.dropout_rate = checkpoint['dropout_rate']

    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)

    model = get_model_from_args(args, args.model, args.num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# CHANGE
def save_vit_detector(args, model, optimizer, epoch):
    saved_model_name = args.model + "_" + args.model_arch + "_" + str(args.image_size) + args.method + "_id_" +\
                       args.dataset + "_ood_" + args.ood_dataset + "_bs" + str(args.batch_size) + "_best_accuracy.pth"
    model_path = "saved_models/trained_detector/"

    # create a second file, indicating how many epochs have passed until the best accuracy was reached
    with open(model_path + args.dataset + '.txt', 'w') as f:
        f.write(str(epoch) + ":     " + args.model + "_" + args.model_arch + "_" + str(args.image_size) + args.method + "_" + args.dataset\
                       + "_bs" + str(args.batch_size) + "_best_accuracy.pth")

    torch.save({
        'model_name': args.model, # args.model,
        'loss': args.method,  # args.loss, #args.method,
        'dataset': args.dataset,
        'ood_dataset': args.ood_dataset,
        'num_classes': args.num_classes,

        'eps': args.eps,
        'norm': args.norm,
        'iterations': args.iterations,
        'restarts': args.restarts,
        'stepsize': args.stepsize,
        'noise': args.noise,

        'image_size': args.image_size, # args.img_size, # args.image_size,
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,

        'emb_dim': args.emb_dim, # 768
        'mlp_dim': args.mlp_dim, # 3072
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'attn_dropout_rate': args.attn_dropout_rate,
        'dropout_rate': args.dropout_rate,

        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

    }, model_path + saved_model_name)
    print("Model saved in path: ", model_path + saved_model_name)


def get_train_detector_args():
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic args
    parser.add_argument("--model", type=str, default="vit", help="model used to detect ood samples")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str, help="ViT pretrained model type")
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['t16', 'vs16', 's16', 'b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument('--eval', action='store_true', help='evaluate on dataset')
    parser.add_argument('--opt', default='SGD', type=str, choices=('AdamW', 'SGD'))
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")

    parser.add_argument('--device', type=str, default="cuda", help='str - cpu or cuda to calculate the tensors on')
    #parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use") # NO PARALLELIZATION with attack so only one gpu is possible
    parser.add_argument("--select-gpu", type=int, default=0, help="specific gpu to use, because parallelization is not possible")
    parser.add_argument("--tensorboard", default=False, action='store_true', help='flag of turnning on tensorboard')
    #parser.add_argument("--classifier", type=str, default="vit", help="model used to classify id samples")
    parser.add_argument("--classifier-ckpt-path", type=str, default=None, help="model checkpoint to load weights")

    parser.add_argument("--image-size", type=int, default=384, help="input image size", choices=[128, 160, 224, 384])
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--train-steps", type=int, default=10000, help="number of training/fine-tunning steps")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
    parser.add_argument("--warmup-steps", type=int, default=500, help='learning rate warm up steps')

    parser.add_argument("--data-dir", type=str, default='data/cifar10/', help='id data folder')
    parser.add_argument("--dataset", type=str, default='cifar10', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--ood-data-dir", type=str, default='data/', help='ood data folder')
    parser.add_argument("--ood-dataset", type=str, default='svhn', help="out-of-distribution dataset")
    #parser.add_argument("--num-classes", type=int, default=2, help="number of classes should be two for ID and OOD")


    # * Mixup params
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0)')  # later we can try it wd >0
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.0, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # similarity metric
    parser.add_argument('--sim_metric', type=str, default='Cosine', choices=['Cosine', 'Euclidean', 'Mahalanobis'], help='similarity metric used in contrastive loss')
    parser.add_argument('--method', type=str, default='SupCon', help='for SupCon,SimCLR etc')
    parser.add_argument('--head', type=str, default=None, help='for contrastive head and linear head')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')

    # settings for the PGD attack
    parser.add_argument('--eps', type=float, default=0.01, help='float - the radius of the max perturbation ball around the sample')
    parser.add_argument('--norm', type=str, default="inf", help='str - inf or l2 norm (currently only inf)')
    parser.add_argument('--iterations', type=int, default=15, help='int - how many steps of perturbations for each restart')
    parser.add_argument('--restarts', type=int, default=2, help='int - how often the MPGD attack starts over at a random place in its eps-space')
    parser.add_argument('--stepsize', type=float, default=0.01, help='float - factor to change the model weights in gradient descent of the adversarial attack')
    parser.add_argument('--noise', type=str, default="normal", help='str - normal, uniform, contraster or decontraster noise is possible')

    # other settings
    parser.add_argument('--attack', action='store_true', help='toggels the MonotonePGD attack')
    parser.add_argument('--break-early', action='store_true', help='interupt execution earlier for developing purposes')
    parser.add_argument('--contrastive', action='store_true', help='using distributed loss calculations across multiple GPUs')
    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')
    parser.add_argument('--test', action='store_true', help='just to block the GPUs')
    parser.add_argument('--test_contrastive_acc', action='store_true', help='test iterative accuracy accross all epoch for contrastive loss')
    args = parser.parse_args()

    # model args
    args = eval("get_{}_config".format(args.model_arch))(args)

    # CHANGE comment out storing args in directory, only printing is left in
    # process_config(args)
    print_config(args)
    args.num_classes = 2 #is always forced to be 2 for ID and OOD
    return args

if __name__ == '__main__':
    args = get_train_detector_args()
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

    main(args, args.device)

