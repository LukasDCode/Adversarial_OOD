import matplotlib.pyplot as plt
import torch
torch.set_grad_enabled(False)
from torchvision.transforms import transforms as T
from utils.cifar100_labels import cifar100_labels
from utils.torch_to_pil import tensor_to_pil_image


# These mean & std are used to map images from [0;1] to [-1;1]
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
            T.Resize((224, 224)),
            #T.ToTensor(), # image is already a torch tensor, not a PIL Image
            T.Normalize(mean, std),
            #T.Normalize(cifar100_mean, cifar100_std),
        ])
# just resize the image for visualization
resize_transform = T.Compose([T.Resize((224, 224))])

def visualize_batch_attn_embeddings(model, img_batch, label_batch, batch_id, pert=False):
    for index, image in enumerate(img_batch):
        label = cifar100_labels[label_batch[index].item()]
        additional_info = "_" + str(batch_id) + "_" + str(index)
        visualize_attn_embeddings(model, image, label, additional_info, pert)


def visualize_attn_embeddings(model, img, img_label, additional_info, pert=False): # img [3, 224, 224]
    # # Detection - Visualize encoder-decoder multi-head attention weights
    # Here we visualize attention weights of the last decoder layer. This corresponds to visualizing, for each
    # detected objects, which part of the image the model was looking at to predict this specific bounding box and class.

    # We will use hooks to extract attention weights (averaged over all heads) from the transformer.

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, cls_token_attn = [], [], []

    # def get_attension_vit(self, input, output):
    #     enc_attn_weights.append(self.scores)

    hooks = [
        model.transformer.encoder_layers[-2].attn.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ), ]

    if pert:
        # here we can do something special for only the perturbed images, in case we would need that
        pass

    img = transform(img)

    # propagate through the model
    _ = model(img.clone().detach().unsqueeze(0))

    for hook in hooks:
        hook.remove()

    # enc_attn_weights.append(model.backbone.transformer.blocks[-1].attn.scores)

    # don't need the list anymore
    conv_features = conv_features[0]

    for i in range(12):
        cls_token_attn.append(torch.squeeze(model.transformer.encoder_layers[i].attn.score)[:, 0, 1:])
        enc_attn_weights.append(torch.squeeze(model.transformer.encoder_layers[i].attn.score)[:, 1:, 1:])

    im = tensor_to_pil_image(img)

    f_map = (14, 14)
    head_to_show = 8  # change head number based on which attention u want to see

    for i in range(12):
        if i in [4, 11]:  # show only 4th and 11th or last layer
            # get the HxW shape of the feature maps of the ViT
            shape = f_map
            # and reshape the self-attention to a more interpretable shape
            cls_attn = cls_token_attn[i][head_to_show].reshape(shape)
            # print(np.around(cls_attn,2))
            # print('Shape of cls attn : ',cls_attn.shape)
            sattn = enc_attn_weights[i][head_to_show].reshape(shape + shape)
            # print("Reshaped self-attention:", sattn.shape)
            # print('Showing layer {} and and head {}'.format(i,head_to_show))

            fact = 16  # as image size was 160 and number of attn block 160/16=10

            # let's select 3 reference points AND CLASIFICATION TOKEN for visualization in transformed image space
            idxs = [(0, 0), (48, 48), (96, 96), (144, 144), ]

            # here we create the canvas
            fig = plt.figure(constrained_layout=True, figsize=(25 * 0.9, 12 * 0.9))
            # and we add one plot per reference point
            gs = fig.add_gridspec(2, 4)
            axs = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[0, -1]),
                fig.add_subplot(gs[1, -1]),
            ]

            # for each one of the reference points, let's plot the self-attention
            # for that point
            for idx_o, ax in zip(idxs, axs):
                if idx_o == (0, 0):
                    idx = (idx_o[0] // fact, idx_o[1] // fact)
                    ax.imshow(cls_attn, cmap='cividis', interpolation='nearest')
                    ax.axis('off')
                    # ax.set_title(f'cls attn at layer: {i}')
                    ax.set_title('cls token attention', fontsize=22)
                else:
                    idx = (idx_o[0] // fact, idx_o[1] // fact)
                    ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
                    ax.axis('off')
                    # ax.set_title(f'self-attn{idx_o} at layer: {i}')
                    ax.set_title(f'self-attention at{idx_o}', fontsize=22)

            # and now let's add the central image, with the reference points as red circles
            fcenter_ax = fig.add_subplot(gs[:, 1:-1])
            fcenter_ax.imshow(resize_transform(im))  # cls_attn
            for (y, x) in idxs:
                if not (x == 0 and y == 0):
                    x = ((x // fact) + 0.5) * fact
                    y = ((y // fact) + 0.5) * fact
                    fcenter_ax.add_patch(plt.Circle((x, y), fact // 3, color='r'))
                    fcenter_ax.axis('off')
            if pert:
                plt.savefig(f'figures/vit_attention/{img_label}{additional_info}_att-layer{i}_pert.png')
            else:
                plt.savefig(f'figures/vit_attention/{img_label}{additional_info}_att-layer{i}_clean.png')
            plt.close('all')

