import torch
import torch.nn as nn
import torch.nn.functional as F


import utils.models.modules_ibp as modules_ibp




# PyTorch models inherit from torch.nn.Module
class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) #only one output because of BCELoss

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x







# copied from Alex Code
class CNN_IBP(nn.Module):
    def __init__(self, dset_in_name='CIFAR10', num_classes=2, last_layer_neg=False):
        super().__init__()
        if dset_in_name == 'MNIST':
            self.color_channels = 1
            self.hw = 28
            num_classes = 10 if num_classes is None else num_classes
        elif dset_in_name == 'CIFAR10' or dset_in_name == 'SVHN':
            self.color_channels = 3
            self.hw = 32
            num_classes = 10 if num_classes is None else num_classes
        elif dset_in_name == 'CIFAR100':
            self.color_channels = 3
            self.hw = 32
            num_classes = 100 if num_classes is None else num_classes
        elif dset_in_name == 'RImgNet':
            self.color_channels = 3
            self.hw = 224
            num_classes = 9 if num_classes is None else num_classes
        else:
            raise ValueError(f'{dset_in_name} dataset not supported.')
        self.num_classes = num_classes

        if last_layer_neg:
            last_layer_type = modules_ibp.LinearI_Neg
        else:
            last_layer_type = modules_ibp.LinearI
        self.last_layer_type = last_layer_type


        self.width = 1
        self.C1 = modules_ibp.Conv2dI(self.color_channels, 128 * self.width, 3, padding=1, stride=1)
        self.A1 = modules_ibp.ReLUI()
        self.C2 = modules_ibp.Conv2dI(128 * self.width, 256 * self.width, 3, padding=1, stride=2)
        self.A2 = modules_ibp.ReLUI()
        self.C3 = modules_ibp.Conv2dI(256 * self.width, 256 * self.width, 3, padding=1, stride=1)
        self.A3 = modules_ibp.ReLUI()
        self.pool = modules_ibp.AvgPool2dI(2)
        self.F = modules_ibp.FlattenI()
        self.L4 = modules_ibp.LinearI(256 * self.width * (self.hw//4)**2, 128)
        self.A4 = modules_ibp.ReLUI()
        self.L5 = last_layer_type(128, self.num_classes)

        self.layers = (self.C1,
                       self.A1,
                       self.C2,
                       self.A2,
                       self.C3,
                       self.A3,
                       self.pool,
                       self.F,
                       self.L4,
                       self.A4,
                       self.L5,
                       )

        self.__name__ = f'CNN_S-{self.width}_' + dset_in_name

    def forward(self, x):
        x = x.type(torch.get_default_dtype())
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_layer(self, x, idx):
        x = x.type(torch.get_default_dtype())
        for layer in self.layers[:idx]:
            x = layer.forward(x)
        return x

    def forward_layer_list(self, x, layers):
        bs = x.shape[0]
        output = []
        x = x.type(torch.get_default_dtype())
        for idx, layer in enumerate(self.layers[:max(layers) + 1]):
            if idx in layers:
                output.append(x.view(bs, -1))

            x = layer.forward(x)

        return torch.cat(output, 1)

    def ibp_forward(self, l, u):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers:
            l, u = layer.ibp_forward(l, u)
        return l, u

    def ibp_forward_layer(self, l, u, idx):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers[:idx]:
            l, u = layer.ibp_forward(l, u)
        return l, u

    def ibp_elision_forward(self, l, u, num_classes):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers[:-1]:
            l, u = layer.ibp_forward(l, u)

        layer = self.layers[-1]
        assert isinstance(layer, modules_ibp.LinearI)
        W = layer.weight
        Wd = W.unsqueeze(dim=1).expand((num_classes, num_classes, -1)) - W.unsqueeze(dim=0).expand(
            (num_classes, num_classes, -1))
        ud = torch.einsum('abc,nc->nab', Wd.clamp(min=0), u) + torch.einsum('abc,nc->nab', Wd.clamp(max=0), l)
        if layer.bias is not None:
            bd = layer.bias.unsqueeze(dim=1).expand((num_classes, num_classes)) - layer.bias.unsqueeze(
                dim=0).expand((num_classes, num_classes))
            ud += bd.unsqueeze(0)

        if layer.bias is not None:
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() + layer.bias[:, None]).t()
        else:
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()).t()
        l, u = l_, u_
        return l, u, ud

    def make_table(self):
        sl = []
        for l in self.layers:
            pass

    def forward_pre_logit(self, x):
        x = x.type(torch.get_default_dtype())
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return self.layers[-1](x), x.detach().cpu()

    #     def rescale(self):
    #         EPS = 1e-8
    #         scale1 = self.layers[13].weight.data.abs().mean(1) + EPS
    #         scale2 = self.layers[15].weight.data.abs().mean(0) + EPS

    #         r = (scale2 / scale1 ) ** .5

    #         w1 = self.layers[13].weight.data * r[:,None]
    #         b1 = self.layers[13].bias.data * r
    #         w2 = self.layers[15].weight.data / r[None,:]

    #         self.layers[13].weight.data = w1
    #         self.layers[15].weight.data = w2
    #         self.layers[13].bias.data = b1

    #         scale1 = self.layers[11].weight.data.abs().mean(1) + EPS
    #         scale2 = self.layers[13].weight.data.abs().mean(0) + EPS

    #         r = (scale2 / scale1 + 1e-8) ** .5

    #         w1 = self.layers[11].weight.data * r[:,None]
    #         b1 = self.layers[11].bias.data * r
    #         w2 = self.layers[13].weight.data / r[None,:]

    #         self.layers[11].weight.data = w1
    #         self.layers[13].weight.data = w2
    #         self.layers[11].bias.data = b1

    def rescale(self):
        s = []
        layer_idx = [0, 2, 4, 6, 8, 11, 13, 15]
        for idx in layer_idx[:-1]:
            a = self.layers[idx].weight.data.abs().mean()
            s.append(a.log().item())
        s = torch.tensor(s)

        s8 = self.layers[15].weight.data.abs().mean().log().item()
        fudge_factor = 1.
        beta = fudge_factor * (s.sum() - 7 * s8) / 8

        for i, idx in enumerate(layer_idx[:-1]):
            self.layers[idx].weight.data *= (-beta / 7).exp()
            self.layers[idx].bias.data *= (-(i + 1) * beta / 7).exp()
        self.layers[15].weight.data *= beta.exp()
