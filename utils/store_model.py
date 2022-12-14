import time

import torch

from utils.get_model import get_model_from_args


def save_model(args, classification_model, optimizer):

    #CHANGE
    args.model = "vit"

    # Save the model
    #saved_model_name = args.model + "_" + str(args.image_size) + "SupCE_" + args.dataset + "_bs" \
    #                   + str(args.batch_size / 2) + "_epochs" + str(args.epochs) + "_" + str(int(time.time())) + ".pth"
    saved_model_name = args.model + "_" + str(args.image_size) + "SupCE_" + args.dataset + "_bs" \
                       + str(args.batch_size) + "TEST.pth"
    # saved_model_name = "test.pth"

    try:
        _ = args.data_id
        is_classification_model = False
        model_path = "utils/models/saved_models/detector/"
    except AttributeError:
        is_classification_model = True
        model_path = "utils/models/saved_models/classifier/"

    if is_classification_model:
        dataset = args.dataset.lower()
    else:
        dataset = args.data_id.lower()

    num_classes = 100 if dataset == "cifar100" else 10

    torch.save({
        'model_name': args.model, # args.model,
        'image_size': args.image_size, # args.img_size, # args.image_size,
        'dataset': dataset,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        #'lr': args.lr,
        #'epoch': args.epochs,
        'model_state_dict': classification_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': args.method, # args.loss, #args.method,
    }, model_path + saved_model_name)
    print("Model saved in path: ", model_path + saved_model_name)



def load_classifier(args):
    if args.classification_ckpt:  # args.classification_ckpt #checkpoint_path
        checkpoint_path = args.classification_ckpt  # args.classification_ckpt
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


def load_detector(args):
    if args.detector_ckpt_path:  # args.classification_ckpt #checkpoint_path
        checkpoint_path = args.detector_ckpt_path  # args.classification_ckpt
    else:
        raise ValueError("No checkpoint path for the classifier was specified")

    checkpoint = torch.load(checkpoint_path)

    args.model = checkpoint['model_name']
    args.method = checkpoint['loss']
    args.dataset = checkpoint['dataset']
    args.ood_dataset = checkpoint['ood_dataset']
    args.num_classes = checkpoint['num_classes'] # 10 or 100

    # try except, because checkpoint from the 7.12. does not contain these properties
    try:
        args.method = checkpoint['method']
        args.contrastive = checkpoint['contrastive']
    except:
        args.method = "SimCLR"
        args.contrastive = False

    args.image_size = checkpoint['image_size']
    args.batch_size = checkpoint['batch_size'] # 64
    args.patch_size = checkpoint['patch_size'] # 16

    args.eps = checkpoint['eps']
    args.norm = checkpoint['norm']
    args.iterations = checkpoint['iterations']
    args.restarts = checkpoint['restarts']
    args.stepsize = checkpoint['stepsize']
    args.noise = checkpoint['noise']

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
    


