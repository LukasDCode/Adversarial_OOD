from torchvision.transforms import transforms as T

def tensor_to_pil_image(img):
    image = img.view(img.size(0), -1)
    image -= image.min(1, keepdim=True)[0]
    image /= image.max(1, keepdim=True)[0]
    image = image.view(3, 224, 224)
    return T.ToPILImage(mode='RGB')(image)

