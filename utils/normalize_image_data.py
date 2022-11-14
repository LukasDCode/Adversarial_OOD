import torchvision

# image normalization is done to get to the "standard score" https://en.wikipedia.org/wiki/Standard_score
def normalize_cifar10_image_data(image_data):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(cifar10_mean, cifar10_std)])
    return transform(image_data)

def normalize_cifar100_image_data(image_data):
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(cifar100_mean, cifar100_std)])
    return transform(image_data)

def normalize_SVHN_image_data(image_data):
    svhn_mean = (0.4377, 0.4438, 0.4728)
    svhn_std = (0.1980, 0.2010, 0.1970)
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(svhn_mean, svhn_std)])
    return transform(image_data)

def normalize_general_image_data(image_data):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean, std)])
    return transform(image_data)