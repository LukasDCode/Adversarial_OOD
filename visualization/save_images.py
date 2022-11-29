from torchvision.transforms import transforms as T
from utils.cifar100_labels import cifar100_labels
from utils.torch_to_pil import tensor_to_pil_image

def save_img_batch_as_jpgs(image_batch, label_tensor, batch_index, perturbed=False):
    # https://stackoverflow.com/questions/63733998/how-do-you-output-images-from-a-batchdataset-of-images-in-keras
    for index, label in enumerate(label_tensor):
        image = image_batch[index]
        pil_image = tensor_to_pil_image(image)

        # create 100 cifar folders "$ mkdir cifar100_{0..99}
        #pil_image._save("pgd_visualization/cifar100_" + str(label.item()) + "/" + str(batch_index) + "_" + str(index) + additional_info + ".jpg")

        if perturbed:
            # Naming of images: Original Label _ inter batch index _ intra batch index
            pil_image.save("figures/pgd_visualization/" + str(cifar100_labels[label.item()]) + "_" + str(batch_index) + "_" + str(index) + "_pert.png")
        else:
            pil_image.save("figures/pgd_visualization/" + str(cifar100_labels[label.item()]) + "_" + str(batch_index) + "_" + str(index) + "_clean.png")


def save_img_as_jpg(image, label, batch_index, perturbed=False):
    pil_image = tensor_to_pil_image(image)
    if perturbed:
        # Naming of images: Original Label _ inter batch index _ intra batch index
        pil_image.save("figures/pgd_visualization/" + str(cifar100_labels[label.item()]) + "_" + str(batch_index) + "_pert.png")
    else:
        # Naming of images: Original Label _ inter batch index _ intra batch index
        pil_image.save("figures/pgd_visualization/" + str(cifar100_labels[label.item()]) + "_" + str(batch_index) + "_clean.png")

