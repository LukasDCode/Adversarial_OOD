#!/usr/bin/env python
# coding: utf-8

# # Visualization of VIT classification
# 
# In this notebook, we show-case how to:
# * use the pretrained models that we provide to make classifications
# * visualize the attentions of the model to gain insights on the way it sees the images.

# ## Preliminaries
# This section contains the boilerplate necessary for the other sections. Run it first.

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')

#sys.path.insert(0, "../..")

import os
#os.chdir("../..")


# In[2]:


import pandas as pd
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from pylab import rcParams

from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

import torch
from torch import nn
from torch.nn import functional as F

#import torchvision.transforms as T
torch.set_grad_enabled(False);


# # HYPERPARAMETERS
# The following cell sets all hyperparams for either cifar10, cifar100 or im30.

# In[3]:

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
    
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

print("Image size:", image_size)
print("Number of classes:", len(classes))
print("Classes:", classes)


# Declare all the params here now....

# In[4]:


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

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# In[5]:


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{classes[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


# # Detection - using a pretrained model from TorchHub
# 
# In this section, we show-case how to load a model from existing checkpoint, run it on a custom image, and print the result.
# Here we load the simplest of vit-160) for fast inference.

# In[6]:


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


# We now retrieve the image as a PIL image

# In[7]:


# get image from img_path and store it in PIL format for further processing
split_text = img_path.split('/')
img_name = split_text[-2]+'_'+split_text[-1].split('.')[0]
dataset_dir_path = "/".join(split_text[:-2])
print("Path to train dataset   :", path_train_imgs)
print("Path to test dataset    :", path_test_imgs)

print("dataset_dir_path:", dataset_dir_path)

print('Opening test image from :', img_path)
print("Image name              :", img_name)


#im = Image.open(img_path)
#print(im)


# In[8]:

"""
# mean-std normalize the input image (batch-size: 1)
img = transform(im)
#img = img.unsqueeze(0).to(device="cpu")#.cuda()
img = img.unsqueeze(0).to(device=device)#.cuda()
print('Shape of the image:  ', img.shape)
# propagate through the model
outputs, softmax_prediction = model(img, feat_cls=True)
print('Shape of output:     ',outputs.shape)
softmax_score = F.softmax(softmax_prediction).data.cpu().numpy()
softmax_score = np.around(softmax_score[0],4)
softmaxt_sort_ind = np.argsort(softmax_score)[::-1]
print("Predicted class:     ",np.asarray(classes)[softmaxt_sort_ind])
softmaxt_sort_score  = np.sort(softmax_score)[::-1]
print("Srt Score:           ",softmaxt_sort_score)
#print('Softmax prediction : ', F.softmax(softmax_prediction).data.cpu())
"""

# In[9]:


def get_model_output_from_single_image(full_img_path):
    img = Image.open(full_img_path)
    print(img)
    img = transform(img)
    print(img)
    print(img.size())
    print(type(img))
    #img = img.unsqueeze(0).to(device="cpu")#.cuda()
    img = img.unsqueeze(0).to(device=device)  # .cuda()
    print(img)
    print(img.size())
    print(len(img[0][1]))
    print(type(img))
    outputs, softmax_prediction = model(img, feat_cls=True)
    return outputs, softmax_prediction
_,_ = get_model_output_from_single_image(img_path)


# # Centroid - find 768-dimensional mean for a given class
# The find_class_centroid() method takes all output tensors from one class and calculates the mean from them. This will be the center of this class, from which we will construct the e-ball and see if pictures are OOD or ID.

# In[10]:


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


# ### Iterate over either the TRAIN or the TEST dataset

# In[11]:


do_break = False
break_after_x_iterations = 50
directory = dataset_dir_path + "/"

def iterate_over_entire_dataset(directory):
    mean_of_classes = {}
    for index, image_class in enumerate(classes): # iterates 10 classes
        list_of_outputs = []
        data_dir = directory + image_class + "/"
        #print("data_dir", data_dir)
        break_counter = break_after_x_iterations
        for file in os.listdir(data_dir): # iterates 5k images of same class
            if do_break and break_counter == 0: break
            break_counter -= 1
            full_img_path = os.path.join(data_dir, file)
            outputs, _ = get_model_output_from_single_image(full_img_path)
            list_of_outputs.append(outputs)
            #print("Outputs:", outputs)
            #print("List of outputs:", list_of_outputs[0][0])
        #print("Len of outputs:", len(list_of_outputs))
        mean_of_classes[index] = find_class_centroid(list_of_outputs)
        #break
    return mean_of_classes


# In[12]:


# first iterate over train dataset
#mean_of_train_classes = iterate_over_entire_dataset(path_train_imgs)

#print("mean_of_train_classes", mean_of_train_classes)
#print("Shape of last outputs tensor:", outputs.size(), type(outputs))
#print("output", outputs)


# In[13]:


# _save pickle into file for imagenet
pickle_file_path = '/home-local/koner/lukas/Adversarial_OOD/data/'
pickle_file_name = 'cifar10_train_centroids.pickle'
#with open(pickle_file_path + pickle_file_name, 'wb') as handle:
#    pickle.dump(mean_of_train_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[14]:


# second iterate over test dataset
#mean_of_test_classes = iterate_over_entire_dataset(path_test_imgs)

#print("mean_of_test_classes", mean_of_test_classes)
#print("Shape of last outputs tensor:", outputs.size(), type(outputs))
#print("output", outputs)


# In[15]:


# _save pickle into file for imagenet
pickle_file_path = '/home-local/koner/lukas/Adversarial_OOD/data/'
pickle_file_name_test = 'cifar10_test_centroids.pickle'
#with open(pickle_file_path + pickle_file_name_test, 'wb') as handle:
#    pickle.dump(mean_of_test_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[16]:


with open(pickle_file_path + pickle_file_name, 'rb') as handle:
    pickle_content = pickle.load(handle)
print(pickle_content)
mean_of_test_classes = pickle_content


# # Get model output for 1 OOD image

# In[17]:


ood_data_path = '/home-local/koner/lukas/Adversarial_OOD/data/random_ood_images/'
ood_file_name = "chair1.jpg"

ood_output, softmax_prediction = get_model_output_from_single_image(ood_data_path + ood_file_name)

#print('Shape of output: ',ood_output.shape)
softmax_score = F.softmax(softmax_prediction).data.cpu().numpy()
softmax_score = np.around(softmax_score[0],4)
softmaxt_sort_ind = np.argsort(softmax_score)[::-1]
print("Predicted class: ",np.asarray(classes)[softmaxt_sort_ind])
softmaxt_sort_score = np.sort(softmax_score)[::-1]
print("Srt Score:       ",softmaxt_sort_score)

print("\n")
print(ood_output)


# # OOD image distance to Centroids

# In[18]:


# calculate EUCLIDEAN distance from the OOD image to every class
for index in range(len(mean_of_test_classes)):
    euclidean_dist = torch.cdist((mean_of_test_classes[index]).to(device=device), ood_output)**2
    print("Distance:", euclidean_dist.item())
    #euclidean_dist_2 = sum(((mean_of_test_classes[index] - ood_output)**2).reshape(768))
    #print("Distance:", euclidean_dist_2)


# In[19]:


# calculate COSINE distance from the OOD image to every class
for index in range(len(mean_of_test_classes)):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_sim = cos((mean_of_test_classes[index]).to(device=device), ood_output)
    print("Cosine Similarity:", cosine_sim.item())


# ### Output and distance of an ID image

# In[20]:


# image of a ship, class with index 8 (0 indicated) --> smallest distance and biggest cosine similarity
id_output, _ = get_model_output_from_single_image('/home-local/koner/lukas/contrastive_ood/data/cifar10png/test/ship/0090.jpg') # contrastive_ood doesnt exist anymore :(
for index in range(len(mean_of_test_classes)):
    euclidean_dist = torch.cdist((mean_of_test_classes[index]).to(device=device), id_output)**2
    print("Distance         :  ", euclidean_dist.item())
    cos_id = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_sim_id = cos((mean_of_test_classes[index]).to(device=device), id_output)
    print("Cosine Similarity:  ", cosine_sim_id.item())


# # Copy OOD image to another path

# """
# import shutil
# 
# # copy the clean image to the path where the images get adversarially altered
# os.chdir(ood_data_path)
# dst_dir = '/home/koner/adversarial_ood/data/adversarially_altered_images/'
# for img in os.listdir():
#     if img == ood_file_name:
#         #print(img)
#         shutil.copy(img, dst_dir)
# """

# ## TO DO
# - Read in images from dataset in the vector representation --> dataloader or whatever
# - Do simple PGD
# - start adding features on top of PGD
#     - Adaptive Step-size
#     - Backtracking
#     - Normalize Gradients
#     - Momentum
#     - ... etc.
# - Step 4
# - Profit

# # READ CIFAR100 Dataset

# In[21]:


cifar100_path = "/home-local/koner/lukas/Adversarial_OOD/data/cifar-100-python/"
#function to read files present in the Python version of the dataset
def unpickle(file):
    with open(cifar100_path + file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict


# In[22]:


trainData = unpickle('train')
#type of items in each file
for item in trainData:
    print(item, type(trainData[item]))
print(len(trainData['data']))
print(len(trainData['data'][0]))
print(np.unique(trainData['fine_labels']))
print(np.unique(trainData['coarse_labels']))
print(trainData['batch_label'])


# #### Fine Labels = specific labels of 100
# #### Coarse Labels = superordinate labels of 20
# #### Every coarse label includes about 5 different fine labels

# In[23]:


testData = unpickle('test')
metaData = unpickle('meta')
#metaData
print("Fine labels:", metaData['fine_label_names'], "\n")
print("Coarse labels:", metaData['coarse_label_names'])


# In[24]:


#storing coarse labels along with its number code in a dataframe
category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
#storing fine labels along with its number code in a dataframe
subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])
#print(category)
#print(subCategory)


# In[25]:


X_train = trainData['data']
print(X_train)
print(type(X_train))
print(len(X_train))
print()
print(len(X_train[0]))
print(len(X_train[49999]))
print(X_train[49999])


# In[26]:


#4D array input for building the CNN model using Keras
X_train = X_train.reshape(len(X_train),3,32,32).transpose(0,2,3,1)
#X_train


# In[27]:


X_train


# In[28]:


#generating a random number to display a random image from the dataset along with the label's number and name
#setting the figure size
rcParams['figure.figsize'] = 2,2
#generating a random number
imageId = np.random.randint(0, len(X_train))
#showing the image at that id
plt.imshow(X_train[imageId])
#setting display off for the image
plt.axis('off')
#displaying the image id
print("Image number selected : {}".format(imageId))
#displaying the shape of the image
print("Shape of image : {}".format(X_train[imageId].shape))
#displaying the category number
print("Image category number: {}".format(trainData['coarse_labels'][imageId]))
#displaying the category name
print("Image category name: {}".format(category.iloc[trainData['coarse_labels'][imageId]][0].capitalize()))
#displaying the subcategory number
print("Image subcategory number: {}".format(trainData['fine_labels'][imageId]))
#displaying the subcategory name
print("Image subcategory name: {}".format(subCategory.iloc[trainData['fine_labels'][imageId]][0].capitalize()))


# In[29]:


#transforming the testing dataset
X_test = testData['data']
X_test = X_test.reshape(len(X_test),3,32,32).transpose(0,2,3,1)
y_train = trainData['fine_labels']
y_test = testData['fine_labels']

#number of classes in the dataset
n_classes = 100
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


# # Projected Gradient Descent
# Since the optimization objective in Eq. (1) is non-convex, all of our attacks
# are basically just heuristics. Nonetheless, some approaches do tend to work
# much better than others in practice and these approaches tend to have a lot of
# justification behind them within the convex framework. I highly recommend
# skimming at least some of Boyd’s Convex Optimization.
# 1PGD: Let’s start writing down the basic algorithm for l ∞ -PGD and then add
# in elements that have worked well in the past. Assume we are minimizing some
# loss L : R d → R:
# 
# x̂ (i+1) = x (i) − α · ∇L(x (i) ),
# 
# x (i+1) = Proj(x̂ (i+1)).
# 
# The projection Proj : R d → [0, 1] d finds the closest point to x̂ (i+1) that still lies
# in the intersection of our ball B e (x (0) ) and the range of valid images [0, 1] d . In
# principle, this is a convex optimization problem that one would have to solve
# on each step. However, it has a very simple closed form solution. For each
# coordinate of x̂ you can simply compute.
# 
# This looks more complicated than it is. You can compute this in a vectorized
# way by first clamping each coordinate to lie within the e − l ∞ -box and then
# the [0, 1]-box separately (use the torch.clamp-function). And that’s projected
# gradient descent!

# In[30]:


def get_model_output_from_single_image_ood(img_as_nparray):
    #print(img_as_nparray.shape)
    img_as_nparray = img_as_nparray.transpose(2,0,1)
    #img_as_tensor = torch.from_numpy(img_as_nparray).to(device="cpu")#.cuda()
    #img_as_tensor = torch.from_numpy(img_as_nparray).to(device=device)#.cpu()

    #print(img_as_tensor.size())
    #pil_image = T.ToPILImage(mode='RGB')(img_as_tensor)
    pil_image = T.ToPILImage(mode='RGB')(img_as_nparray)

    #print(pil_image)
    transformed_img = transform(pil_image)
    transformed_img = transformed_img.unsqueeze(0).to(device=device)
    #print(transformed_img)
    outputs, softmax_prediction = model(transformed_img, feat_cls=True)#.to(device="cpu")
    return outputs, softmax_prediction


# In[31]:

ood_outputs, ood_softmax = get_model_output_from_single_image_ood(X_train[0])
# In[32]:


#print(ood_outputs.size())
#print(ood_softmax)


# In[33]:


from utils.ood_detection.PGD_attack import NormalNoiseGenerator
from utils.ood_detection.PGD_attack import MonotonePGD, MaxConf, MonotonePGD_trial
from vit.src.data_loaders import create_dataloaders
from utils.dotdict import dotdict

# In[34]:


dataloaders_config = {
    "data_dir": cifar100_path, # "/home-local/koner/lukas/Adversarial_OOD/data/cifar-100-python/"
    "image_size": 224,
    "batch_size": 16,
    "num_workers": 0,
    "contrastive": False,
    "albumentation": True,
    "net": "vit",
    "no_train_aug": False,
    "dataset": "cifar100", # in_dataset
    "deit": False    
}
dataloaders_config = dotdict(dataloaders_config)


# In[35]:


# args copy pasted from a debug session
args = {
	'dset_in_name': 'CIFAR10',
	'dset_out_name': 'CIFAR100',
	'architecture': {
		'dset_in_name': 'CIFAR10',
		'num_classes': 10,
		'detector_path': 1,
		'arch_style': 'CNN',
		'arch_size': 'S',
		'file_path': None,
		'use_last_bias': False,
		'last_layer_neg': False
	},
	'train': {
		'schedule': {
			'lr_schedule_type': 'constant',
			'lr_schedule': [0.1, 0.1, 0.01, 0.001, 0.0001],
			'lr_schedule_epochs': [1, 50, 75, 90, 100],
			'kappa_schedule_type': 'constant',
			'kappa_schedule': [0.0, 0.01],
			'kappa_schedule_epochs': [10, 100],
			'weight_decay': 0.0005,
			'eps_schedule_type': 'constant',
			'eps_schedule': [0.01],
			'eps_schedule_epochs': [100]
		},
		'train_type': 'plain',
		'in_loss_type': 'binary',
		'out_loss_type': 'binary',
		'expfolder': None,
		'tb_folder': 'default',
		'batch_size': 512,
		'use_adam': False,
		'eps': 0.01,
		'momentum': 0.9
	},
	'tb_name': {
		'string_base': None,
		'hp_base': {
			'type': 'train.train_type',
			'S': 'architecture.arch_size',
			'L': 'train.in_loss_type',
			'ds': 'dset_in_name'
		},
		'string_extra': None,
		'hp_extra': None
	},
	'gpu': 0,
	'redirect': True,
	'detect_anomaly': False,
	'augmentation': {
		'train_exclude': 'H',
		'autoaugment': False,
		'hflip': False,
		'crop': 4
	},
	'eval': {
		'eps': 0.01,
		'batch_size': 100
	}
}


# In[36]:


#device = "cpu"
train_dataloader, valid_dataloader = create_dataloaders(dataloaders_config)

eps = args['train']['eps'] # 0.01
iterations = 3
stepsize = 0.01
norm = "inf"
num_classes = args['architecture']['num_classes'] # 10
from_logits = True
loss = "CrossEntropy"
restarts = 2 #5
lr_decay = 0.5
lr_gain = 1.1

PGD_args = dotdict({
    "type": "PGD",
    "model": model,
    "eps": eps,
    "iterations": iterations,
    "stepsize": stepsize,
    "norm": norm,
    "num_classes": num_classes,
    "loss": loss,
    "restarts": restarts,
    "lr_gain": lr_gain,
    "lr_decay": lr_decay
})


# In[37]:


"""
if num_classes==1:
    reduction = lambda x: torch.sigmoid(x)
else:
    reduction = ( lambda x: torch.softmax(x, dim=1).max(1)[0] ) if from_logits else (lambda x: x.exp().max(1)[0])
    #reduction = (lambda x: x.exp().max(1)[0])
if 'use_last_class' in args and args['use_last_class']: # not used --> jump to else case
    use_last_class = True
    reduction = ( lambda x:  - torch.log_softmax(x, dim=1)[:,-1] )
    apgd_loss = 'last_conf'
    loss = attacks.LastConf()
else:
    use_last_class = False
    apgd_loss = 'max_conf'
    loss = MaxConf(from_logits)
"""

reduction = ( lambda x: torch.softmax(x, dim=1).max(1)[0] ) if from_logits else (lambda x: x.exp().max(1)[0])
use_last_class = False
loss = MaxConf(from_logits)


# In[38]:




noise = NormalNoiseGenerator(sigma=1e-4)
#noise = UniformNoiseGenerator(min=-eps, max=eps)
#noise = Contraster(eps)
#noise = DeContraster(eps)
#attack = MonotonePGD(eps, iterations, stepsize, num_classes, model=PGD_args.model)

alex_attack = False #True #False
if alex_attack:
    attack = MonotonePGD(eps, iterations, stepsize, num_classes, momentum=0.9, norm=norm, loss=loss,
                         normalize_grad=False, early_stopping=0, restarts=PGD_args.restarts,
                         init_noise_generator=noise, model=PGD_args.model, save_trajectory=False)
else:
    attack = MonotonePGD_trial(eps, iterations, stepsize, num_classes, momentum=0.9, norm=norm, loss=loss,
                               normalize_grad=False, early_stopping=0, restarts=PGD_args.restarts,
                               init_noise_generator=noise, model=PGD_args.model, save_trajectory=False)




# In[39]:


train_dataloader
print(train_dataloader)
print(type(model))


# ### The following code cell explained as good as possible
# Every batch contains 16 images (after 16 batches there is a break --> 256 images)  
# x are 16 ood images, y are the 16 correct class labels, x = [16, 3, 224, 224] (224 = 14 * 16)  
#   
# - `clean_conf` are the 16 prediction scores the model returns for the given batch of 16 images
# - `altered_image_batch` is the tensor representation of all 16 images of the batch after they have been adversarially altered  
# - `attacked_point` are the 10 prediction scores for each of the 10 classes of the model for a batch of 16 altered images
# - `reduced_attacked_point` returns the max (in regards to all classes) of the softmax of the `attacked point` resulting in the 16 highest softmax scores for the 16 images  
# - again `attacked_point` **??!!??**
# - `o` is the softmax (from the reduction) of the resulting predictions / class predictions
# - `out` is a list storing the `clean_conf`, the `reduced_attacked_point` and `o`  
# - `best` is a list with the highest predictions for each image, but because all of these images are OOD images the predictions should be quite low across the board. This list gets parsed to a torch tensor at the very end

# In[40]:


best = []
for batch_idx, (x, y) in enumerate(tqdm(train_dataloader)):
    
    #if batch_idx==args.batches:
    if batch_idx == dataloaders_config.batch_size:
        break # breaks after 16 iterations
        
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        # reduction = torch.softmax(x).max(1)[0]
        # reduction only returns the max values of the softmax values
        # https://discuss.pytorch.org/t/what-is-the-meaning-of-max-of-a-variable/14745
        clean_conf = reduction( model(x) ).detach().cpu() # size = 16

    out = [ clean_conf ]
    
    """
    #if args.dataset!='RImgNet':
    if dataloaders_config.dataset != "RImgNet":
    
        #attacked_point = model(attack.perturb(x.clone(), y)[0]).detach().cpu()
        altered_image_batch = attack.perturb(x.clone(), y) # size = [16, 3, 224, 224]
        attacked_point = model(altered_image_batch).detach().cpu() # size = [16, 10]
        
        reduced_attacked_point = reduction(attacked_point) # size = 16
        out.append(reduced_attacked_point)
    """

    # calls the perturb() of RestartAttack --> calls perturb_inner() of MonotonePGD
    attacked_point = attack(x.clone(), y) # size = [16, 3, 224, 224]
    
    o = model(attacked_point) # size = [16, 10]
    o = reduction( o ).detach().cpu() # size = 16
    out.append( o )

    max_conf, att_idx = torch.stack(out, dim=0).max(0) # both have size = 16 (where values of att_idx={0,1,2})

    best.append(max_conf)
    break
best = torch.cat(best, 0)


# In[41]:


print(best.size())
print(best)


# maybe good to read:
# https://towardsdatascience.com/know-your-enemy-7f7c5038bdf3
# 
# at 12min:
# https://www.youtube.com/watch?v=VyWAvY2CF9c


# For Visualization of Embeddings and Attention the cells from the other notebook have to be copied here. They have been deleted to keep this notebook as straight forward as possible.
