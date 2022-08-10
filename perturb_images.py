import torch
from vit.src.model import VisionTransformer as ViT

from visualization.save_images import save_img_batch_as_jpgs, save_img_as_jpg
from visualization.visualize_embeddings import visualize_attn_embeddings, visualize_batch_attn_embeddings
from utils.cifar100_labels import cifar100_labels
from utils.normalize_image_data import normalize_image_data
from utils.image_modification_path import print_single_image_modification_path, print_batch_image_modification_path
from utils.calculate_softmax_score import calculate_softmax_score


def perturb_image_batch(model, attack, dataloader, device, visualization=False):
    """
    returns 2 lists consisting of X numpy arrays with 10 entries
    1st list is the clean softmax prediction regarding all 10 cifar10 classes of the unperturbed image
    2nd list is the perturbed softmax prediction regarding all 10 cifar10 classes of the perturbed image
    """
    clean_out = []
    perturbed_out = []
    for batch_idx, (image_batch, label_batch) in enumerate(dataloader):

        if visualization:
            save_img_batch_as_jpgs(image_batch.clone().detach(), label_batch, batch_idx, perturbed=False)
            visualize_batch_attn_embeddings(model, image_batch.clone().detach(), label_batch, batch_idx, pert=False)

        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        with torch.no_grad():
            normalized_image_batch = normalize_image_data(image_batch.clone().detach())
            if type(model) == type(ViT()):
                softmax_prediction = model(normalized_image_batch, feat_cls=False) # feat_cls=True also returns embeddings
            else:
                softmax_prediction = model(image_batch)  # size = [16, 10]

            clean_out.extend(calculate_softmax_score(softmax_prediction, model.classifier.out_features))

        # calls the perturb() of RestartAttack --> which calls perturb_inner() of MonotonePGD
        perturbed_image_batch, best_softmax_list, best_idx = attack(image_batch.clone(), label_batch)  # size = [16, 3, 224, 224]
        normalized_perturbed_image_batch = normalize_image_data(perturbed_image_batch.clone().detach())

        print_batch_image_modification_path(best_softmax_list, best_idx, model.classifier.out_features, label_batch)

        if type(model) == type(ViT()):
            softmax_prediction = model(normalized_perturbed_image_batch, feat_cls=False) # feat_cls=True also returns embeddings
        else:
            softmax_prediction = model(perturbed_image_batch)  # size = [16, 10]

        perturbed_out.extend(calculate_softmax_score(softmax_prediction, model.classifier.out_features))

        if visualization:
            save_img_batch_as_jpgs(perturbed_image_batch.clone().detach(), label_batch, batch_idx, perturbed=True)
            visualize_batch_attn_embeddings(model, perturbed_image_batch.clone().detach(), label_batch, batch_idx, pert=True)
        break  # breaks after 1 batch
    return clean_out, perturbed_out


def perturb_single_image(model, attack, dataloader, device, visualization=False):
    """
    returns 2 lists consisting of X numpy arrays with 10 entries
    1st list is the clean softmax prediction regarding all 10 cifar10 classes of the unperturbed image
    2nd list is the perturbed softmax prediction regarding all 10 cifar10 classes of the perturbed image
    """
    clean_out = []
    perturbed_out = []
    for batch_idx, (image_batch, label_batch) in enumerate(dataloader): # image_batch [16, 3, 224, 224]

        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        with torch.no_grad():
            for index, image in enumerate(image_batch): # image [3, 224, 224]
                label = cifar100_labels[label_batch[index].item()] #string format of label ex: "mountain"

                normalized_image = normalize_image_data(image.clone().detach().unsqueeze(0)) # [-2.4;2.7]
                if visualization:
                    save_img_as_jpg(image.clone().detach(), label_batch[index], batch_idx, perturbed=False)
                    visualize_attn_embeddings(model, image.clone().detach(), label, "", pert=False)

                if type(model) == type(ViT()):
                    softmax_prediction = model(normalized_image, feat_cls=False) # feat_cls=True also returns embeddings
                    # normalized values [-2.4;2.7]
                else:
                    softmax_prediction = model(image.unsqueeze(0)) # has to be [1, 3, 224, 224]

                clean_out.extend(calculate_softmax_score(softmax_prediction, model.classifier.out_features))
                # calls the perturb() of RestartAttack --> calls perturb_inner() of MonotonePGD
                perturbed_image, best_softmax_list, best_idx = attack(image.clone().unsqueeze(0), label_batch[index])  # size = [1, 3, 224, 224]
                normalized_perturbed_image = normalize_image_data(perturbed_image.clone().detach())

                # some debugging info, can be deleted #DELETE TODO
                # default pytorch floating point precision is 4 digits after the point
                # "shamelessly taken from NumPy" - https://pytorch.org/docs/stable/generated/torch.set_printoptions.html
                if attack.eps >= round((image - perturbed_image).abs().max().item(), 6) > attack.eps * 0.9: #eps = 0.01
                    print("+++++ Attack is within eps range (max:", (image - perturbed_image).abs().max().item(), ") +++++")

                print_single_image_modification_path(best_softmax_list[best_idx.item()], model.classifier.out_features, label)

                if type(model) == type(ViT()):
                    softmax_prediction = model(normalized_perturbed_image, feat_cls=False) # feat_cls=True also returns embeddings
                else:
                    softmax_prediction = model(perturbed_image)  # size = [16, 10]

                perturbed_out.extend(calculate_softmax_score(softmax_prediction, model.classifier.out_features))

                if visualization:
                    save_img_as_jpg(perturbed_image.clone().detach()[0], label_batch[index], batch_idx, perturbed=True)
                    visualize_attn_embeddings(model, perturbed_image.clone().detach().squeeze(0), label, "", pert=True)
                break  # break after 1 image
        break  # break after 1 batch
    return clean_out, perturbed_out

