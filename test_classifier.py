import sys
import argparse
from tqdm import tqdm
import torch
import torchvision
import numpy as np

from vit.src.model import VisionTransformer as ViT
from vit.src.utils import MetricTracker, accuracy
from utils.ood_detection.ood_detector import MiniNet, CNN_IBP
from utils.load_data import get_test_dataloader
from utils.store_model import load_classifier


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


def test_classifier(args):
    classifier = load_classifier(args)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    metrics = MetricTracker(*[metric for metric in metric_names], writer=None)
    log = {}
    losses = []
    acc1s = []
    acc5s = []

    # get a dataloader mixed 50:50 with ID and OOD data and labels of 0 (ID) and 1 (OOD)
    test_dataloader = get_test_dataloader(args)

    with torch.no_grad():
        classifier.eval()
        criterion = torch.nn.CrossEntropyLoss().to(args.device)  # CHANGE appended to device and placed outside of loop
        running_test_error = 0
        for epoch_nr, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            # from ViT training validation
            metrics.reset()
            inputs, labels = inputs.to(device=args.device), labels.to(device=args.device)
            outputs = classifier(inputs)

            running_test_error += error_criterion(outputs.squeeze(1), labels)
            if args.device == "cuda": torch.cuda.empty_cache()



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
    print("\nOld metrics")
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
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--n-gpu", type=int, default=2, help="number of gpus to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print("num_workers is set to 0 in debugging mode, otherwise no useful debugging possible")
        args.num_workers = 0

    test_classifier(args)

    print("finished all executions")


