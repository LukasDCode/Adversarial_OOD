import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from vit.src.model import VisionTransformer
from vit.src.config import get_train_config
from vit.src.checkpoint import load_checkpoint
from vit.src.data_loaders import create_dataloaders
from vit.src.utils import setup_device, accuracy, MetricTracker, TensorboardWriter,write_json
from util import adjust_learning_rate
from losses import SupConLoss
from OOD_Distance import run_ood_distance


def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu'),
                contrastive=False, test_contrastive_acc=False, method=None, criterion2=None, head=None, mixup_fn=None):
    metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader)):
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
            batch_pred, pred_classifier = model(batch_data, not_contrastive_acc = not test_contrastive_acc)
        else:
            # batch_data values are in range [0;1]
            batch_pred = model(batch_data, not_contrastive_acc=not test_contrastive_acc)

        if contrastive:
            bsz = batch_target.shape[0]
            f1, f2 = torch.split(batch_pred, [bsz, bsz], dim=0)
            batch_pred = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if method == 'SimCLR':
            loss = criterion(batch_pred)
        else:
            loss = criterion(batch_pred, batch_target)

        if head=="both" and contrastive:# train both head contrastive and classifier
            loss1 =  criterion2(pred_classifier, batch_target)
            loss = loss + 0.2*loss1
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        #torch.cuda.empty_cache()
        if metrics.writer is not None:
            metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        if mixup_fn is  None: # for mixup dont calculate accuracy
            if  batch_idx % 100 == 10 and not contrastive:
                acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
                metrics.update('acc1', acc1.item())
                metrics.update('acc5', acc5.item())

            if batch_idx % 100 == 10 and not test_contrastive_acc:
                print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, Acc@5: {:.2f}"
                        .format(epoch, batch_idx, len(data_loader), loss.item(), 0 if contrastive else acc1.item(),0 if contrastive else acc5.item()))#, acc5.item()
    return metrics.result()


def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    criterion = torch.nn.CrossEntropyLoss()
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data, eval=args.eval)
            loss = criterion(batch_pred, batch_target)
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    if metrics.writer is not None:
        metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('acc5', acc5)
    return metrics.result()


# CHANGE Currently completely commented out in the code
def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False, save_freq=100):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'ckpt_epoch_current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'ckpt_epoch_best.pth')
        torch.save(state, filename)
    elif epoch%save_freq==0:
        filename = str(save_dir + 'ckpt_epoch_' + str(epoch) + '.pth')
        print('Saving file : ',filename)
        torch.save(state, filename)


def main(args, device, device_ids):

    # tensorboard
    if args.tensorboard:
        if not args.test:
            writer = TensorboardWriter(args.summary_dir, args.tensorboard)
        else:
            writer = None
    else:
        writer = None

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    model = VisionTransformer(
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
        if not args.test_contrastive_acc and not args.eval and args.num_classes != state_dict['classifier.weight'].size(0):#not
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print("re-initialize fc layer")
            missing_keys = model.load_state_dict(state_dict, strict=False)
        else:
            missing_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys from checkpoint ",missing_keys.missing_keys)
        print("Unexpected keys in network : ",missing_keys.unexpected_keys)


    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    args.model = 'vit'
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
        optimizer = torch.optim.AdamW(params=model.parameters(),lr=args.lr,weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            momentum=0.9)
    if args.cosine:
        lr_scheduler=None
    else:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr,
            pct_start=args.warmup_steps / args.train_steps,
            total_steps=args.train_steps)


    # start training
    print("start training")
    best_acc = 0.0
    best_epoch = 0
    args.epochs = args.train_steps // len(train_dataloader)
    print("length of train loader : ",len(train_dataloader),' and total epoch ',args.epochs)
    for epoch in range(1, args.epochs + 1):
        if args.cosine:
            adjust_learning_rate(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            print("learning rate at {0} epoch is {1}".format(epoch, param_group['lr']))

        log = {'epoch': epoch}

        if not args.eval:
            # train the model
            model.train()
            result = train_epoch(epoch, model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics, device,
                                 contrastive=args.contrastive, test_contrastive_acc=args.test_contrastive_acc, method=args.method,
                                 head=args.head, criterion2=criterion2, mixup_fn = mixup_fn)
            log.update(result)

        # validate the model
        if not args.contrastive:
            model.eval()
            result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, device)
            log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        if args.test_contrastive_acc or args.eval or not args.contrastive:
            if log['val_acc1'] > best_acc:
                best_acc = log['val_acc1']
                best_epoch = epoch

                # CHANGE
                # save the model with the best accuracy
                print("save the model with the best accuracy")
                save_vit_model(args, model, optimizer, epoch)



        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))

    if args.test_contrastive_acc or args.eval or not args.contrastive:
        print("Best accuracy : ",best_acc, ' for ',best_epoch)# saving class mean
        best_curr_acc = {'best_acc': best_acc,'best_epoch': best_epoch,
                         'curr_acc': log['val_acc1'],'curr_epoch': epoch}
        if args.tensorboard:
            write_json(best_curr_acc,os.path.join(args.checkpoint_dir,'acc.json'))


# CHANGE
def save_vit_model(args, model, optimizer, epoch):

    #CHANGE currently only vit supported
    args.model = "vit"

    # Save the model
    saved_model_name = args.model + "_" + args.model_arch + "_" + str(args.image_size) + args.method + "_" + args.dataset\
                       + "_bs" + str(args.batch_size) + "_best_accuracy.pth"

    model_path = "saved_models/trained_classifier/"

    """
    # create a second file, indicating how many epochs have passed until the best accuracy was reached
    with open(model_path + args.dataset + '.txt', 'w') as f:
        f.write(str(epoch) + ":     " + args.model + "_" + args.model_arch + "_" + str(args.image_size) + args.method + "_" + args.dataset\
                       + "_bs" + str(args.batch_size) + "_best_accuracy.pth")
    """

    torch.save({
        'model_name': args.model, # args.model,
        'loss': args.method,  # args.loss, #args.method,
        'dataset': args.dataset,
        'num_classes': args.num_classes,

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


if __name__ == '__main__':
    args = get_train_config()
    device, device_ids = setup_device(args.n_gpu)
    main(args, device, device_ids)
    print("finished all executions")

