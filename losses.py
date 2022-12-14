"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

from OOD_Distance import euclidean_dist, mahalanobis


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, similarity_metric='Cosine'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.sim_metric = similarity_metric

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if self.sim_metric == 'Euclidean':
            eps = 0.00005
            euc_anchor_contrast = - torch.div(euclidean_dist(anchor_feature, contrast_feature) + eps, self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(euc_anchor_contrast, dim=1, keepdim=True)
            logits = euc_anchor_contrast - logits_max.detach()

        elif self.sim_metric == 'Mahalanobis':
            eps = 0.00005
            assert batch_size > 10
            in_classes = torch.unique(labels[:batch_size])
            class_idx = [torch.nonzero(cls == labels[:batch_size]).squeeze(dim=1)[:, 0] for cls in in_classes]
            classes_feats = [contrast_feature[idx] for idx in class_idx]
            classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats], dim=0)
            sup_inv_cov = [np.linalg.inv(np.cov(cls_feats.cpu().detach().numpy(), rowvar=False)) for cls_feats in classes_feats]

            # CHANGE changed to torch.tensor
            maha_anchor_contrast = - torch.div(mahalanobis(anchor_feature, classes_mean, sup_inv_cov) + eps, self.temperature) ## fixme mismatch between np and tensor variables
            #maha_anchor_contrast = - torch.div(mahalanobis(anchor_feature, classes_mean, torch.FloatTensor(sup_inv_cov)) + eps, self.temperature)

            # for numerical stability
            logits_max, _ = torch.max(maha_anchor_contrast, dim=1, keepdim=True)
            logits = maha_anchor_contrast - logits_max.detach()

        else:
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        #CHANGE added "+ 1e-6" to the end
        #log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
