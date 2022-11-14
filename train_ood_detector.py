from tqdm import tqdm
import time
import argparse
import torch
import torchvision

from utils.normalize_image_data import normalize_general_image_data
from utils.ood_detection.load_data import shuffle_batch_elements
from utils.ood_detection.ood_detector import MiniNet, CNN_IBP
from ood_detection.load_data import get_mixed_test_dataloader, get_mixed_train_valid_dataloaders
from vit.src.model import VisionTransformer as ViT

from PGD_Alex import MonotonePGD, MaxConf
from PGD_Alex import UniformNoiseGenerator, NormalNoiseGenerator, Contraster, DeContraster


def get_model_from_args(args, model_name, num_classes):
    if model_name.lower() == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device=args.device) # cuda()
    elif model_name.lower() == "vit":
        #"""
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


def get_noise_from_args(args):
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
    return noise


def train_detector(args, classification_model):
    if args.device == "cuda": torch.cuda.empty_cache()

    detector_model = get_model_from_args(args, args.detector_model_name, num_classes=2)
    # TODO maybe different loss for vit model
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(detector_model.parameters(), lr=args.lr, momentum=args.momentum)

    attack = MonotonePGD(args.eps, args.iterations, args.stepsize, num_classes=2, momentum=0.9, norm=args.norm,
                         loss=MaxConf(True), normalize_grad=False, early_stopping=0, restarts=args.restarts,
                         init_noise_generator=get_noise_from_args(args), model=classification_model, save_trajectory=False)

    mixed_train_dataloader, mixed_valid_dataloader = get_mixed_train_valid_dataloaders(args)

    epoch_number = 0
    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        detector_model.train(True)
        # TODO train one epoch normally and one epoch with adversarially altered images
        avg_loss = train_one_epoch(args, detector_model, attack, loss_fn, optimizer, mixed_train_dataloader)
        # We don't need gradients on to do reporting
        detector_model.train(False) # same as model.eval()

        with torch.no_grad():
            #Error calculation
            running_valid_error = 0
            running_vloss = 0.0
            for i, (vdata_id, vdata_ood) in enumerate(mixed_valid_dataloader):
                vinputs, vlabels = shuffle_batch_elements(vdata_id, vdata_ood)
                vinputs, vlabels = vinputs.to(device=args.device), vlabels.to(device=args.device)
                normalized_vinputs = normalize_general_image_data(vinputs) # once WAS vinputs.clone()
                voutputs = detector_model(normalized_vinputs)
                vloss = loss_fn(voutputs.squeeze(1), vlabels)
                running_vloss += vloss.item()
                running_valid_error += error_criterion(voutputs.squeeze(1), vlabels)
                if args.device == "cuda": torch.cuda.empty_cache()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            #Error
            avg_valid_error = running_valid_error / (i+1)
            print("Average Validation Error:", avg_valid_error.item())

        epoch_number += 1

    if args.save_model:
        # Save the model
        model_path = "utils/models/saved_models/"
        saved_model_name = args.detector_model_name + "_" + str(args.img_size) + "SupCE_ID" + args.data_id + "_OOD"\
                           + args.data_ood + "_bs" + str(args.batch_size) + "_lr" + str(args.lr).strip(".") + "_epochs"\
                           + str(args.epochs) + "_" + str(int(time.time())) + ".pth"
        #torch.save(detector_model.state_dict(), model_path+saved_model_name)
        torch.save({
            'model_name': args.detector_model_name,
            'img_size': args.img_size,
            'data_id': args.data_id,
            'data_ood': args.data_ood,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epoch': args.epochs,
            'model_state_dict': detector_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': args.loss,
        }, model_path + saved_model_name)
        print("Model saved in path: ", model_path+saved_model_name)


def train_one_epoch(args, detector_model, attack, loss_fn, optimizer, mixed_train_dataloader):
    running_loss, running_attack_loss, last_loss, last_attack_loss = 0., 0., 0., 0.

    for i, (data_id, data_ood) in enumerate(tqdm(mixed_train_dataloader)):
        inputs, labels = shuffle_batch_elements(data_id, data_ood)
        inputs, labels = inputs.to(device=args.device), labels.to(device=args.device)
        inputs.requires_grad = True #possible because leaf of the acyclic graph
        normalized_inputs = normalize_general_image_data(inputs.clone()) # no detach because requires gradient
        # clone is needed because inputs are later used again for perturbation
        # inputs:              [batch_size, channels, img_size, img_size]
        # normalized_inputs:   [batch_size, channels, img_size, img_size]

        optimizer.zero_grad() # Zero gradients for every batch
        outputs = detector_model(normalized_inputs)
        loss = loss_fn(outputs.squeeze(1), labels) # Compute the loss and its gradients
        loss.backward()
        optimizer.step() # Adjust learning weights
        running_loss += loss.item() # Gather data and report
        del normalized_inputs, loss # delete for performance reasons to free up cuda memory
        if args.device == "cuda": torch.cuda.empty_cache()



        # TODO include perturbation training here
        # Here also the ID data gets perturbed, but actually it is just an augmentation
        # calls the perturb() of RestartAttack --> which calls perturb_inner() of MonotonePGD
        perturbed_inputs, _, _ = attack(inputs, labels) # once WAS inputs.clone()
        perturbed_normalized_inputs = normalize_general_image_data(perturbed_inputs) # once WAS perturbed_inputs.clone()
        # perturbed_inputs:            size = [16, 3, 224, 224]
        # perturbed_normalized_inputs: size = [16, 3, 224, 224]

        optimizer.zero_grad()  # Zero gradients for every batch
        outputs = detector_model(perturbed_normalized_inputs)
        loss = loss_fn(outputs.squeeze(1), labels)  # Compute the loss and its gradients
        loss.backward()
        optimizer.step()  # Adjust learning weights
        running_attack_loss += loss.item()  # Gather data and report
        del inputs, perturbed_inputs, perturbed_normalized_inputs, loss  # delete for performance reasons to free up cuda memory
        if args.device == "cuda": torch.cuda.empty_cache()


        print_interval = 100
        if i % print_interval == print_interval-1:
            last_loss = running_loss / print_interval  # loss per batch
            last_attack_loss = running_attack_loss / print_interval # attack loss per batch
            print("   Batch", i+1, "loss:", last_loss, "attack_loss:", last_attack_loss)
            running_loss, running_attack_loss = 0., 0.

    return last_loss


def error_criterion(outputs,labels):
    """
    used to calculate the errors in the validation phase
    """
    prediction_tensor = torch.where(outputs>0., 1., 0.)
    train_error = (prediction_tensor != labels).float().sum()/prediction_tensor.size()[0]
    return train_error


def test_detector(args):
    model_path = "utils/models/saved_models/"
    #saved_model_name = args.detector_model_name + "_" + str(args.img_size) + "SupCE_ID" + args.data_id + "_OOD"\
    #                       + args.data_ood + "_bs" + str(args.batch_size) + "_lr" + str(args.lr).strip(".") + "_epochs"\
    #                       + str(args.epochs) + "_" + str(int(time.time())) + ".pth"

    saved_model_name = "vit_224SupCE_IDcifar10_OODsvhn_bs16_lr0.0001_epochs120_1667968086.pth"
    #"resnet_224SupCE_IDcifar10_OODsvhn_bs32_lr0.0001_epochs8_1667912039.pth"

    # Maybe some adjustments in the 'if' condition, if not running with the args
    model = get_model_from_args(args, args.detector_model_name, num_classes=2)

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
            if args.device == "cuda": torch.cuda.empty_cache()

            # TODO add perturbation to testing AFTER it fully works in training

        # Error
        avg_valid_error = running_test_error / (i + 1)
        print("Average Test Error:", avg_valid_error.item())
        print("Finished Testing the Model")


def parse_args():
    parser = argparse.ArgumentParser(description='Run the monotone PGD attack on a batch of images, default is with ViT and the MPGD of Alex, where cifar10 is ID and cifar100 is OOD')

    parser.add_argument('--classification_model_name', type=str, default="vit", help='str - what model should be used to classify input samples "vit", "resnet", "mininet" or "cnn_ibp"')
    parser.add_argument('--detector_model_name', type=str, default="vit",
                        help='str - what model should be used to detect id vs ood "vit", "resnet", "mininet" or "cnn_ibp"')
    parser.add_argument('--class_ckpt', type=str,
                        default="/nfs/data3/koner/contrastive_ood/save/vit/vit_224SupCE_cifar10_bs512_lr0.01_wd1e-05_temp_0.1_210316_122535/checkpoints/ckpt_epoch_50.pth",
                        help='str - path of pretrained model checkpoint')
    parser.add_argument('--det_ckpt', type=str, default="/nfs/data3/koner/contrastive_ood/save/vit/vit_224SupCE_cifar10_bs512_lr0.01_wd1e-05_temp_0.1_210316_122535/checkpoints/ckpt_epoch_50.pth", help='str - path of pretrained model checkpoint')
    parser.add_argument('--device', type=str, default="cuda", help='str - cpu or cuda to calculate the tensors on')
    parser.add_argument('--data_id', type=str, default="cifar10", help='str - the in-distribution dataset "cifar10", "cifar100" or "svhn"')
    parser.add_argument('--data_ood', type=str, default="svhn", help='str - the out-distribution dataset "cifar10", "cifar100" or "svhn"')
    parser.add_argument('--loss', type=str, default="bce", help='str - how the loss is calculated for the ood sample "bce" ("maxconf" not working yet)')

    parser.add_argument('--eps', type=float, default=0.01, help='float - the radius of the max perturbation ball around the sample')
    parser.add_argument('--norm', type=str, default="inf", help='str - inf or l2 norm (currently only inf)')
    parser.add_argument('--iterations', type=int, default=15, help='int - how many steps of perturbations for each restart')
    parser.add_argument('--restarts', type=int, default=2, help='int - how often the MPGD attack starts over at a random place in its eps-space')
    parser.add_argument('--noise', type=str, default="normal", help='str - normal, uniform, contraster or decontraster noise is possible')

    parser.add_argument('--epochs', type=int, default=4, help='int - amount of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='float - learning rate of the model')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='float - how much the lr drops after every unsuccessfull step')
    parser.add_argument('--lr_gain', type=float, default=1.1, help='float - how much the lr raises after every successfull step')
    parser.add_argument('--stepsize', type=float, default=0.01, help='float - factor to change the model weights in gradient descent of the adversarial attack')
    parser.add_argument('--num_heads', type=int, default=4, help='int - amount of attention heads for the vit model')
    parser.add_argument('--num_layers', type=int, default=5, help='int - amount of parallel layers doing the calculations for the vit model')
    parser.add_argument('--momentum', type=float, default=0.9, help='float - factor to change the model weights in gradient descent')

    parser.add_argument('--img_size', type=int, default=224, help='int - amount of pixel for the images')
    parser.add_argument('--batch_size', type=int, default=128, help='int - amount of images in the train, valid or test batches')
    parser.add_argument('--workers', type=int, default=0, help='int - amount of workers in the dataloader')

    # boolean parameters, false until the flag is set --> then true
    parser.add_argument('--visualize', action='store_true', help='flag to store the original & perturbed images and the attention maps as png files')
    parser.add_argument('--test', action='store_true', help='flag to set testing to true and test a model')
    parser.add_argument('--save_model', action='store_true', help='flag to save the model if it is being finished training')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    classification_model_ckpt = torch.load(args.class_ckpt, map_location=torch.device(args.device))
    classification_model = get_model_from_args(args, args.classification_model_name, num_classes=10)
    classification_model.load_state_dict(classification_model_ckpt['state_dict'], strict=False)
    classification_model = classification_model.to(device=args.device)  # cuda()
    classification_model.eval()


    #"""
    args.epochs = 2
    args.lr = 0.0001
    args.batch_size = 256 # apparently only 128 is possible, otherwise CUDA out of memory --> RuntimeError

    args.iterations = 3
    args.restarts = 1

    args.save_model = False # True
    args.test = False # if True --> Testing, else False --> Training
    args.detector_model_name = "mininet" # mininet, vit, resnet, cnn_ibp

    #args.img_size = 112
    #"""



    if args.test:
        test_detector(args)
    else:
        try:
            train_detector(args, classification_model)
        except RuntimeError as error:
            print("Cuda Memory Summary:", torch.cuda.memory_summary(device=None, abbreviated=False))
            print("RuntimeeError:", error)



    print("finished all executions")

    # TODO
    # 1. set all labels of cifar10 to 0 (for ID) and cifar100 to 1 (for OOD)
    # 2. throw them into a model (CNN, ResNet, ViT)
    # 3. Train a model with the new labels
    # 3.a) check if all models work with training, saving, loading & testing
    # 4. make perturbations on them and see if the model still detects them correctly
    # 5. switch out the models and train another model


    """
    Restnet (ID cifar10, OOD SVHN)
    works pretty well with
    8 Epochs, 0.0001 lr, 32 Batch size
    
    ViT
    8 Epochs, 0.0001 lr, 32 Batch size --> 9.4% Error rate
    3 Epochs, 0.0001 lr, 64 Batch size --> 9.9% Error rate
    6 Epochs, 0.0001 lr, 64 Batch size --> 10% Error rate
    120 Epochs, 0.0001 lr, 16 Batch size --> 9.2% Error rate
    
    Mininet
    2 Epochs, 0.0001 lr, 16 Batch size --> seems to work pretty well, no numbers so far
    
    CNN IBP
    """

