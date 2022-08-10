import torchvision

# image normalization is done to get to the "standard score" https://en.wikipedia.org/wiki/Standard_score
def normalize_image_data(image_data):
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(cifar100_mean, cifar100_std)])
    return transform(image_data)
