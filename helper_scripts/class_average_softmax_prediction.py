import os

import numpy as np
from PIL import Image
import pickle

from tqdm import tqdm

import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)

#from .vit.src.data_loaders import create_dataloaders
from vit.src.data_loaders import create_dataloaders
from perturb_images import calculate_softmax_score
from utils.dotdict import dotdict


# # HYPERPARAMETERS
# The following cell sets all hyperparams for either cifar10, cifar100 or im30.

device = "cuda" #"cpu"
dataset = 'cifar10' #cifar10 #cifar100 #'im30'
if dataset == 'cifar10':
    image_size = 224
    num_heads= 12
    num_layers= 12
    f_map_size = 14 # = 224/16 = image_size / patch_size
    num_classes = 10
    ckpt = '/nfs/data3/koner/contrastive_ood/_save/vit/vit_224SupCE_cifar10_bs512_lr0.01_wd1e-05_temp_0.1_210316_122535/checkpoints/ckpt_epoch_50.pth'
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # home/wiss/ and contrastive_ood do not exist anymore
    #img_path = '/home/wiss/koner/contrastive_ood/data/cifar10png/test/truck/0071.jpg'
    #img_path = '/home/wiss/koner/contrastive_ood/data/cifar10png/test/ship/0090.jpg'
    img_path = '/home/wiss/koner/contrastive_ood/data/cifar10png/test/bird/0295.jpg'
    path_train_imgs = '/home/wiss/koner/contrastive_ood/data/cifar10png/train/'
    path_test_imgs = '/home/wiss/koner/contrastive_ood/data/cifar10png/test/'
elif dataset == 'cifar100':
    image_size = 224
    num_heads= 12
    num_layers= 12
    f_map_size = 14 # = 224/16 = image_size / patch_size
    num_classes = 100
    ckpt = '/nfs/data3/koner/contrastive_ood/_save/vit/vit_224SupCon_cifar100_bs928_lr0.01_wd1e-05_temp_0.1_wdbothCE_SupCon/checkpoints/ckpt_epoch_100.pth'
    classes = [    
            "beaver", "dolphin", "otter", "seal", "whale",
            "aquarium fish", "flatfish", "ray", "shark", "trout",
            "orchids", "poppies", "roses", "sunflowers", "tulips",
            "bottles", "bowls", "cans", "cups", "plates",
            "apples", "mushrooms", "oranges", "pears", "sweet" "peppers",
            "clock", "computer keyboard", "lamp", "telephone", "television",
            "bed", "chair", "couch", "table", "wardrobe",
            "bee", "beetle", "butterfly", "caterpillar", "cockroach",
            "bear", "leopard", "lion", "tiger", "wolf",
            "bridge", "castle", "house", "road", "skyscraper",
            "cloud", "forest", "mountain", "plain", "sea",
            "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
            "fox", "porcupine", "possum", "raccoon", "skunk",
            "crab", "lobster", "snail", "spider", "worm",
            "baby", "boy", "girl", "man", "woman",
            "crocodile", "dinosaur", "lizard", "snake", "turtle",
            "hamster", "mouse", "rabbit", "shrew", "squirrel",
            "maple", "oak", "palm", "pine", "willow",
            "bicycle", "bus", "motorcycle", "pickup truck", "train",
            "lawn-mower", "rocket", "streetcar", "tank", "tractor"]

    # home/wiss/ and contrastive_ood do not exist anymore
    img_path = '/home/wiss/koner/contrastive_ood/data/cifar100png/test/vehicles_1/pickup_truck/0004.png'
    #img_path = '/home/wiss/koner/contrastive_ood/data/cifar100png/test/flowers/tulip/0097.png'
    #img_path = '/home/wiss/koner/contrastive_ood/data/cifar100png/test/people/woman/0027.png'
    path_train_imgs = '/home/wiss/koner/contrastive_ood/data/cifar100png/train/vehicles_1/pickup_truck/0004.png'
    path_test_imgs = '/home/wiss/koner/contrastive_ood/data/cifar100png/test/vehicles_1/pickup_truck/0004.png'
elif dataset == 'im30':
    image_size = 160
    num_heads= 12
    num_layers= 12
    f_map_size = 10 # = 160/16 = image_size / patch_size
    num_classes = 30
    ckpt = '/nfs/data3/koner/contrastive_ood/_save/vit/vit_160_alb_im30_SupCE_ImageNet_bs256_lr0.01_wd1e-05_temp_0.1_210425_005746/checkpoints/ckpt_epoch_30.pth'
    classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
                'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover', 'mosque', 'nail',
                'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile', 'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
    
    img_path = '/nfs/data3/koner/contrastive_ood/data/ImageNet30/val/airliner/15.JPEG'
    #img_path = '/nfs/data3/koner/contrastive_ood/data/ImageNet30/val/barn/12.JPEG'
    #img_path = '/nfs/data3/koner/contrastive_ood/data/ImageNet30/val/volcano/n09472597/30.JPEG'
    #img_path = '/nfs/data3/koner/contrastive_ood/data/ImageNet30/val/american_alligator/5.JPEG'
    path_train_imgs = '/nfs/data3/koner/contrastive_ood/data/ImageNet30/val/airliner/15.JPEG'
    path_test_imgs = '/nfs/data3/koner/contrastive_ood/data/ImageNet30/val/airliner/15.JPEG'
else:
    raise ValueError("No valid dataset got specified")


# standard PyTorch mean-std input image normalization
from torchvision.transforms import transforms as T
transform = T.Compose([
            T.Resize((image_size,image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
# just resize the image for visualization
resize_transform = T.Compose([
            T.Resize((224,224))])

from vit.src.model import VisionTransformer as ViT
# load ckpt
#ckpt = torch.load(ckpt, map_location=torch.device("cpu"))
ckpt = torch.load(ckpt, map_location=torch.device(device))
model = ViT(image_size=(image_size,image_size),
             num_heads=num_heads,
             num_layers=num_layers,
             num_classes=num_classes,
             contrastive=False,
             timm=True)
# now load the model
model.load_state_dict(ckpt['state_dict'], strict=False)
#model = model.to(device="cpu")#cuda()
model = model.to(device=device)#cuda()
model.eval()
print('Model loaded....')


dataloaders_config = {
    "data_dir": "/home-local/koner/lukas/Adversarial_OOD/data/cifar10/", # "/home/koner/adversarial_ood/data/cifar-100-python/"
    "image_size": 224,
    "batch_size": 16,
    "num_workers": 0,
    "contrastive": False,
    "albumentation": False,
    "net": "vit",
    "no_train_aug": False,
    "dataset": "cifar10", # in_dataset
    "deit": False
}
dataloaders_config = dotdict(dataloaders_config)
train_dataloader, valid_dataloader = create_dataloaders(dataloaders_config)


def get_model_output_from_single_image(full_img_path):
    img = Image.open(full_img_path)
    img = transform(img)
    img = img.unsqueeze(0).to(device=device)  # .cuda()
    outputs, softmax_prediction = model(img, feat_cls=True)
    return outputs, softmax_prediction


# # Centroid - find 768-dimensional mean for a given class
# The find_class_centroid() method takes all output tensors from one class and calculates the mean from them.
# This will be the center of this class, from which we will construct the e-ball and see if pictures are OOD or ID.
def find_class_centroid(list_of_outputs):
    #print(len(list_of_outputs[0][0]))
    centroid_tensor = [] # first as a list, later casted to tensor
    for index in range(len(list_of_outputs[0][0])): # iterates 768 tensor entries
        average_value = 0
        for output in list_of_outputs: # iterates 5k images of same class
            #print("output:", output[0][index].item())
            average_value += output[0][index].item()
        dimension_mean = average_value / len(list_of_outputs)
        #print("Comparison:", dimension_mean)
        centroid_tensor.append(dimension_mean)
    return torch.FloatTensor(centroid_tensor)[None, :] # put the tensor back in the old format [1, 768]
#centroid_tensor = find_class_centroid(list_of_outputs)
#print("Shape os centroid_tensor:", centroid_tensor.size(), type(centroid_tensor))
#print(centroid_tensor)


def calculate_relative_class_mean(softmax_score_list, class_index):
    accumulated_softmax_value = 0
    counter = 0
    for softmax in softmax_score_list:
        max_value = max(softmax)
        max_value_index = np.where(softmax == max_value)[0][0]
        if max_value_index == class_index:
            counter += 1
            accumulated_softmax_value += softmax[class_index]
            if max_value > 1: print(max_value)
    return accumulated_softmax_value / counter

def calculate_absolute_class_mean(list_of_indices, class_index):
    # will remove every list where the first entry is not equal to the img_index {0,..,9}
    #filtered_indices_list = list(filter(lambda x: x[0] == class_index, list_of_indices))
    #class_accuracy = len(filtered_indices_list) / 1000  # amount of images per class in the testset
    counter = 0
    for index_list in list_of_indices:
        if index_list[0] == class_index:
            counter += 1

    return counter/1000


do_break = False
break_after_x_iterations = 50

def iterate_over_entire_dataset(directory):
    absolute_class_mean = {}
    relative_class_mean = {}
    for img_index, image_class in tqdm(enumerate(classes)): # iterates 10 classes
        list_of_softmax_scores = []
        list_of_indices = []
        data_dir = directory + image_class + "/"
        #print("data_dir", data_dir)
        break_counter = break_after_x_iterations
        for file in os.listdir(data_dir): # iterates 1k images for each class
            if do_break and break_counter == 0: break
            break_counter -= 1
            full_img_path = os.path.join(data_dir, file)
            _, softmax_prediction = get_model_output_from_single_image(full_img_path)

            softmax_score = F.softmax(softmax_prediction, dim=1).data.cpu().numpy()
            softmax_score = np.around(softmax_score[0], 4)
            list_of_softmax_scores.append(softmax_score)

            softmax_sort_ind = np.argsort(softmax_score)[::-1]
            list_of_indices.append(softmax_sort_ind)

        relative_class_mean[img_index] = calculate_relative_class_mean(list_of_softmax_scores, img_index)
        absolute_class_mean[img_index] = calculate_absolute_class_mean(list_of_indices, img_index)
        #break
    return absolute_class_mean, relative_class_mean


def iterate_over_entire_dataset_dataloader(data_loader):
    absolute_class_mean = {}
    relative_class_mean = {}
    list_of_softmax_scores = []
    list_of_indices = []
    for batch_idx, (image_batch, label_batch) in enumerate(tqdm(data_loader)): # iterates 10 classes
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        with torch.no_grad():
            _, softmax_prediction = model(image_batch, feat_cls=True)  # size = [16, 10]
            softmax_score_list = calculate_softmax_score(softmax_prediction, model.classifier.out_features) # 10 = amount of cifar10 classes
            list_of_softmax_scores.extend(softmax_score_list)

            #for softmax_score in list_of_softmax_scores:
            softmax_sort_ind = np.argsort(softmax_score_list)[::-1]
            list_of_indices.extend(softmax_sort_ind)
            #print()

            """
            for score_index, softmax_score in enumerate(softmax_score_list):
            
                list_of_softmax_scores.extend(softmax_score_list)
    
                #TODO turn tensor into list and see if it works with a list
    
    
                #list_of_softmax_scores.append(softmax_score)

                softmax_sort_ind = np.argsort(softmax_score)[::-1]
                list_of_indices.append(softmax_sort_ind)

            """
        #if batch_idx >= 5: break
    print()
    for index in range(model.classifier.out_features):
        relative_class_mean[index] = calculate_relative_class_mean(list_of_softmax_scores, index)
        #absolute_class_mean[index] = calculate_absolute_class_mean(list_of_indices, index)
    #return absolute_class_mean, relative_class_mean
    return relative_class_mean

#for batch_idx, (x, y) in enumerate(tqdm(train_dataloader)):

# absolute is calculated where the final prediction was correct
# relative is the confidence of the correctly predicted classes
# mean_of_test_classes_absolute, mean_of_test_classes_relative = iterate_over_entire_dataset(path_test_imgs)
#mean_of_test_classes_absolute, mean_of_test_classes_relative = iterate_over_entire_dataset_dataloader(valid_dataloader)
mean_of_test_classes_relative = iterate_over_entire_dataset_dataloader(valid_dataloader)

#print("Absolute Mean of Test Classes:", mean_of_test_classes_absolute)
print("Relative Mean of Test Classes:", mean_of_test_classes_relative)
print()

# _save pickle into file for imagenet
# the 10 classes are alphabetically sorted, like in the directory
pickle_file_path = '/home-local/koner/lukas/Adversarial_OOD/data/'
pickle_file_name_abs = 'cifar10_test_absolute_softmax_averages.pickle'
pickle_file_name_rel = 'cifar10_test_relative_softmax_averages.pickle'


#with open(pickle_file_path + pickle_file_name_abs, 'wb') as handle:
#    pickle.dump(mean_of_test_classes_absolute, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open(pickle_file_path + pickle_file_name_rel, 'wb') as handle:
    pickle.dump(mean_of_test_classes_relative, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

with open(pickle_file_path + pickle_file_name_rel, 'rb') as handle:
    pickle_content = pickle.load(handle)
print(pickle_content)
mean_of_test_classes_relative = pickle_content

print("Mean of test classes:")
print(mean_of_test_classes_relative)
