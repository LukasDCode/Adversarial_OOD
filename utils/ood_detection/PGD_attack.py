import torch
import torch.nn as nn
import torch.nn.functional as F
from vit.src.model import VisionTransformer


def create_early_stopping_mask(out, y, conf_threshold, targeted):
    finished = False
    conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
    conf_mask = conf > conf_threshold
    if targeted:
        correct_mask = torch.eq(y, pred)
    else:
        correct_mask = (~torch.eq(y, pred))

    mask = 1. - (conf_mask & correct_mask).float()

    if sum(1.0 - mask) == out.shape[0]:
        finished = True

    mask = mask[(..., ) + (None, ) * 3]
    return finished, mask

def calculate_smart_lr(prev_mean_lr, lr_accepted, lr_decay, iterations, max_lr):
    accepted_idcs = lr_accepted > 0
    if torch.sum(accepted_idcs).item() > 0:
        new_lr = 0.5 * (prev_mean_lr + torch.mean(lr_accepted[lr_accepted > 0]).item())
    else:
        new_lr = prev_mean_lr * ( lr_decay ** iterations )

    new_lr = min(max_lr, new_lr)
    return new_lr

def normalize_perturbation(perturbation, p):
    if p == 'inf':
        return perturbation.sign()
    elif p==2 or p==2.0:
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        pert_normalized = torch.nn.functional.normalize(pert_flat, p=p, dim=1)
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError('Projection only supports l2 and inf norm')

def project_perturbation(perturbation, eps, p):
    if p == 'inf':
        mask = perturbation.abs() > eps
        pert_normalized = perturbation

        # project values back onto eps-Ball
        pert_normalized[mask] = eps * perturbation[mask].sign()
        return pert_normalized
    elif p==2 or p==2.0:
        #TODO use torch.renorm
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        norm = torch.norm(perturbation.view(bs, -1), dim=1) + 1e-10
        mask = norm > eps
        pert_normalized = pert_flat
        pert_normalized[mask, :] = (eps / norm[mask, None]) * pert_flat[mask, :]
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError('Projection only supports l2 and inf norm')

def logits_diff_loss(out, y_oh, reduction='mean'):
    #out: model output
    #y_oh: targets in one hot encoding
    #confidence:
    out_real = torch.sum((out * y_oh), 1)
    out_other = torch.max(out * (1. - y_oh) - y_oh * 100000000., 1)[0]

    diff = out_other - out_real

    return TrainLoss.reduce(diff, reduction)

class AdversarialNoiseGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        #generate noise matching the size of x
        raise NotImplementedError()

class Contraster(AdversarialNoiseGenerator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        eps = self.eps
        s = (x > (1 - eps)).float() + torch.clamp(x * (x <= (1 - eps)).float() - eps, 0, 1)
        return s - x

class DeContraster(AdversarialNoiseGenerator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        diff = torch.clamp(x.mean(dim=(1, 2, 3))[:, None, None, None] - x, -self.eps, self.eps)
        return diff

class NormalNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, sigma=1.0, mu=0):
        super().__init__()
        self.sigma = sigma
        self.mu = mu

    def forward(self, x):
        #return self.sigma * torch.randn_like(x, device="cuda:0") + self.mu
        diff = self.sigma * torch.randn_like(x, device=x.device) + self.mu
        return diff

class UniformNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (self.max - self.min) * torch.rand_like(x) + self.min


class MaxConf(nn.Module):
    def __init__(self, from_logits=True):
        super().__init__()
        self.from_logits = from_logits
        
    def forward(self, x, y, x1, y1, reduction='mean'):
        if self.from_logits:
            out = torch.softmax(y, dim=1)
        else:
            out = y
        out = -out.max(1)[0]
        
        if reduction=='mean':
            return out.mean()
        elif reduction=='none':
            return out
        else:
            print('Error, reduction unknown!')

class Adversarial_attack():
    def __init__(self, loss, num_classes, model=None, save_trajectory=False):
        #loss should either be a string specifying one of the predefined loss functions
        #OR
        #a custom loss function taking 4 arguments as train_loss class
        self.loss = loss
        self.save_trajectory = False
        self.last_trajectory = None
        self.num_classes = num_classes
        if model is not None:
            self.model = model
        else:
            self.model = None

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)

    def set_loss(self, loss):
        self.loss = loss

    def _get_loss_f(self, x, y, targeted, reduction):
        #x, y original data / target
        #targeted whether to use a targeted attack or not
        #reduction: reduction to use: 'sum', 'mean', 'none'
        if isinstance(self.loss, str):
            if self.loss.lower() =='crossentropy':
                if not targeted:
                    l_f = lambda data, data_out: -torch.nn.functional.cross_entropy(data_out, y, reduction=reduction)
                else:
                    l_f = lambda data, data_out: torch.nn.functional.cross_entropy(data_out, y, reduction=reduction )
            elif self.loss.lower() == 'logitsdiff':
                if not targeted:
                    y_oh = torch.nn.functional.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: -logits_diff_loss(data_out, y_oh, reduction=reduction)
                else:
                    y_oh = torch.nn.functional.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: logits_diff_loss(data_out, y_oh, reduction=reduction)
            else:
                raise ValueError(f'Loss {self.loss} not supported')
        else:
            #for monotone pgd, this has to be per patch example, not mean
            l_f = lambda data, data_out: self.loss(data, data_out, x, y, reduction=reduction)

        return l_f

    def get_config_dict(self):
        raise NotImplementedError()

    def get_last_trajectory(self):
        if not self.save_trajectory or self.last_trajectory is None:
            raise AssertionError()
        else:
            return self.last_trajectory

    def __get_trajectory_depth(self):
        raise NotImplementedError()

    def set_model(self, model):
        self.model = model

    def check_model(self):
        if self.model is None:
            raise RuntimeError('Attack model not set')

    def perturb(self, x, y, targeted=False):
        #force child class implementation
        raise NotImplementedError()

class Restart_attack(Adversarial_attack):
    #Base class for attacks that start from different initial values
    #Make sure that they MINIMIZE the given loss function
    def __init__(self, loss, restarts,  num_classes, model=None, save_trajectory=False):
        super().__init__(loss, num_classes, model=model, save_trajectory=save_trajectory)
        self.restarts = restarts

    def perturb_inner(self, x, y, targeted=False):
        #force child class implementation
        raise NotImplementedError()

    def perturb(self, x, y, targeted=False):
        #base class method that handles various restarts
        self.check_model()

        is_train = self.model.training
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False


        restarts_data = x.new_empty((1 + self.restarts,) + x.shape)
        restarts_objs = x.new_empty((1 + self.restarts, x.shape[0]))

        #CHANGE
        restarts_softmaxes = []

        if self.save_trajectory:
            self.last_trajectory = None
            trajectories_shape = (self.restarts,) + (self.__get_trajectory_depth(),) + x.shape
            restart_trajectories = x.new_empty(trajectories_shape)

        for k in range(1 + self.restarts):
            #CHANGE
            #k_data, k_obj, k_trajectory = self.perturb_inner(x, y, targeted=targeted) # data, loss, trajectory(of loss)
            k_data, k_obj, k_trajectory, softmax_trajectory = self.perturb_inner(x, y, targeted=targeted)  # data, loss, trajectory_loss, trajectory_softmax_prediction

            #CHANGE
            restarts_softmaxes.append(softmax_trajectory)

            #k_data, k_obj, k_trajectory = self.perturb_inner(x, y, targeted=True)
            restarts_data[k, :] = k_data
            restarts_objs[k, :] = k_obj
            if self.save_trajectory:
                restart_trajectories[k, :] = k_trajectory

        bs = x.shape[0]
        best_idx = torch.argmin(restarts_objs, 0)
        best_data = restarts_data[best_idx, range(bs), :]

        if self.save_trajectory:
            self.last_trajectory = restart_trajectories[best_idx, :, range(bs), :]

        #reset model status
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = True

        #CHANGE
        #return best_data
        return best_data, restarts_softmaxes, best_idx


class MonotonePGD(Restart_attack):
    """
    Replicated from https://github.com/AlexMeinke/Provable-OOD-Detection/blob/master/utils/adversarial/attacks.py
    """
    def __init__(self, eps, iterations, stepsize, num_classes, momentum=0.9, lr_smart=False, lr_decay=0.5, lr_gain=1.1,
                 norm='inf', loss='CrossEntropy', normalize_grad=False, early_stopping=0, restarts=0,
                 init_noise_generator=None, model=None, save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.norm = norm
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator
        self.lr_smart = lr_smart
        self.prev_mean_lr = stepsize
        self.lr_decay = lr_decay #stepsize decay
        self.lr_gain = lr_gain

    def __get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'PGD'
        dict['eps'] = self.eps
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        dict['restarts'] = self.restarts
        #dict['init_sigma'] = self.init_sigma
        dict['lr_gain'] = self.lr_gain
        dict['lr_decay'] = self.lr_decay
        return dict


    def perturb_inner(self, x, y, targeted=False):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        if self.lr_smart:
            lr_accepted = -1 * x.new_ones(self.iterations, x.shape[0])
            lr = self.prev_mean_lr * x.new_ones(x.shape[0])
        else:
            lr = self.stepsize * x.new_ones(x.shape[0])

        #initialize perturbation
        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)
            pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
            pert = project_perturbation(pert, self.eps, self.norm) # stolen from Alex

        prev_loss = 1e13 * x.new_ones(x.shape[0], dtype=torch.float)

        prev_pert = pert.clone().detach()
        prev_velocity = torch.zeros_like(pert)
        velocity = torch.zeros_like(pert)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        #CHANGE
        softmax_trajectory = []

        for i in range(self.iterations + 1):
            pert.requires_grad_(True)
            with torch.enable_grad():
                data = x + pert
                if type(self.model) == type(VisionTransformer()):
                    out = self.model(data, not_contrastive_acc=True)
                else:
                    out = self.model(data)

                #CHANGE
                softmax_trajectory.append(out.clone()) # tensor(64, 10) (batch_size, #classes)

                loss_expanded = l_f(data, out)
                loss = torch.mean(loss_expanded)
                grad = torch.autograd.grad(loss, pert)[0] #, allow_unused=True)[0] # allow_unused=True
                # ^ autograd.grad() does not work with CUDA parallelization, therefore only one GPU is allowed


            with torch.no_grad():
                loss_increase_idx = loss_expanded > prev_loss # <-- Maximization

                pert[loss_increase_idx, :] = prev_pert[loss_increase_idx, :].clone().detach()
                loss_expanded[loss_increase_idx] = prev_loss[loss_increase_idx].clone().detach()
                prev_pert = pert.clone().detach()
                prev_loss = loss_expanded
                #previous velocity always holds the last accepted velocity vector
                #velocity the one used for the last update that might have been rejected
                velocity[loss_increase_idx, :] = prev_velocity[loss_increase_idx, :]
                prev_velocity = velocity.clone().detach()

                if i > 0:
                    #use standard lr in firt iteration
                    lr[loss_increase_idx] *= self.lr_decay
                    lr[~loss_increase_idx] *= self.lr_gain

                if i == self.iterations:
                    break

                if self.lr_smart:
                    lr_accepted[i, ~loss_increase_idx] = lr[~loss_increase_idx]

                if self.early_stopping > 0:
                    finished, mask = create_early_stopping_mask(out, y, self.early_stopping, targeted)
                    if finished:
                        break
                else:
                    mask = 1.

                #pgd on given loss
                if self.normalize_grad:
                    if self.momentum > 0:
                        #https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                        l1_norm_gradient = 1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
                        velocity = self.momentum * velocity + grad / l1_norm_gradient
                    else:
                        velocity = grad
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * velocity + grad
                    #velocity = self.momentum * velocity + grad if grad is not None else self.momentum * velocity
                    norm_velocity = velocity

                pert = pert - mask * lr[:,None,None,None] * norm_velocity
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint


                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        if self.lr_smart:
            self.prev_mean_lr = calculate_smart_lr(self.prev_mean_lr, lr_accepted, self.lr_decay, self.iterations, 2 * self.eps)

        #CHANGE
        #return data, loss_expanded, trajectory
        return data, loss_expanded, trajectory, softmax_trajectory


class MonotonePGD_trial():
    """
    Self-written PGD attack, fully functional but not more performant
    """
    def __init__(self, eps, iterations, stepsize, num_classes, model=None, momentum=0.9, lr_smart=False,
                 lr_decay=0.5, lr_gain=1.1, norm='inf', loss='CrossEntropy', normalize_grad=False, early_stopping=0,
                 restarts=0, init_noise_generator=None, save_trajectory=False):
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.norm = norm
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator
        self.lr_smart = lr_smart
        self.prev_mean_lr = stepsize
        self.lr_decay = lr_decay #stepsize decay
        self.lr_gain = lr_gain

        self.restarts = restarts

        self.loss = loss
        self.save_trajectory = False
        self.last_trajectory = None
        self.num_classes = num_classes
        if model is None:
            raise RuntimeError("Attack model not set")
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.perturb_image_batch(*args, **kwargs)

    def perturb_image_batch(self, x, y, targeted=False):
        """
        This function includes the perturb of the RestartAttack and also the perturb_inner of the MonotonePGD.
        """

        #perturb --> outer minimization of loss (take samples with the least loss)
        highest_loss_batch = x.new_empty((1 + self.restarts,) + x.shape)
        highest_loss_indices = x.new_empty((1 + self.restarts, x.shape[0]))

        for restart in range(self.restarts +1):
            #perturb_inner --> inner maximization of loss
            loss_function = lambda data, data_out: self.loss(data, data_out, x, y, reduction="none")
            lr = self.stepsize * x.new_ones(x.shape[0])

            pert = self.init_noise_generator(x)
            pert = torch.clamp(x + pert, min=0, max=1) - x  # box constraint
            pert = project_perturbation(pert, self.eps, self.norm)

            initial_excessive_loss = torch.full((16,), 1e13, dtype=torch.float)
            prev_loss = initial_excessive_loss

            prev_pert = pert.clone().detach()
            prev_velocity = torch.zeros_like(pert)
            velocity = torch.zeros_like(pert)

            for iteration in range(self.iterations +1):
                pert.requires_grad_(True)
                with torch.enable_grad():
                    perturbed_batch = x + pert

                    # only the ViT model has the property "not_contrastive_acc" (gradient=None issue)
                    if type(self.model) == type(VisionTransformer()):
                        perturbed_output = self.model(perturbed_batch, not_contrastive_acc=True, eval=False)
                    else:
                        perturbed_output = self.model(perturbed_batch)

                    loss_expanded = loss_function(perturbed_batch, perturbed_output)
                    loss = torch.mean(loss_expanded)
                    all_grads = torch.autograd.grad(loss, pert)
                    grad = all_grads[0] #has no grad_fn it's simply the output of the gradient


                with torch.no_grad():
                    max_loss_mask = loss_expanded > prev_loss

                    pert[max_loss_mask, :] = prev_pert[max_loss_mask, :].clone().detach()
                    loss_expanded[max_loss_mask] = prev_loss[max_loss_mask].clone().detach()
                    prev_pert = pert.clone().detach()
                    prev_loss = loss_expanded

                    # previous velocity always holds the last accepted velocity vector
                    # velocity the one used for the last update that might have been rejected
                    velocity[max_loss_mask, :] = prev_velocity[max_loss_mask, :]
                    prev_velocity = velocity.clone().detach()

                    if iteration > 0:
                        # use standard lr in firt iteration
                        lr[max_loss_mask] *= self.lr_decay
                        lr[~max_loss_mask] *= self.lr_gain

                    # no need to calculate further if already the last iteration
                    if iteration == self.iterations:
                        break

                    velocity = self.momentum * velocity + grad if grad is not None else self.momentum * velocity

                    pert = pert - lr[:, None, None, None] * velocity
                    pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
                    pert = project_perturbation(pert, self.eps, self.norm)

            #return perturbed_batch, loss_expanded


            highest_loss_batch[restart, :] = perturbed_batch #DATA OUTPUT OF PERTURB_INNER
            highest_loss_indices[restart, :] = loss_expanded #LOSS OUTPUT OF PERTURB_INNER


        batch_size = list(x.size())[0] #16
        print(batch_size)
        minimal_loss_indices = torch.argmin(highest_loss_indices, dim=0) # <-- MINIMIZATION
        minimal_loss_batch = highest_loss_batch[minimal_loss_indices, range(batch_size), :]

        # if train else eval
        self.model.eval()

        return minimal_loss_batch


def get_noise_from_args(noise_string, eps):
    """
    get_noise_from_args produces a noise generator

    :noise_string: string of what generator to use
    :eps: maximum allowed perturbation
    :return: noise generator object
    """
    if noise_string.lower() == "normal":
        noise = NormalNoiseGenerator(sigma=1e-4)
    elif noise_string.lower() == "uniform":
        noise = UniformNoiseGenerator(min=-eps, max=eps)
    elif noise_string.lower() == "contraster":
        noise = Contraster(eps)
    elif noise_string.lower() == "decontraster":
        noise = DeContraster(eps)
    else:
        noise = NormalNoiseGenerator(sigma=1e-4)
    return noise
