import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.nn.functional import softplus

import matplotlib.pyplot as plt
import numpy as np

from ..modules import LinearRT, LinearLRT, Conv2dRT, Conv2dLRT

def uncert_regression_gal(img_list: torch.Tensor, reduction: str = 'mean'):
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:,:-1].mean(dim=0, keepdim=True)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    if epi.shape[1] == 3:
        epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'mean':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()

def uncert_classification_kwon(p_hat, var='sum'):
    p_mean = torch.mean(p_hat, dim=0)
    ale = torch.mean(p_hat*(1-p_hat), dim=0)
    epi = torch.mean(p_hat**2, dim=0) - p_mean**2
    if var == 'sum':
        ale = torch.sum(ale, dim=1)
        epi = torch.sum(epi, dim=1)
    elif var == 'top':
        ale = aleatoric[torch.argmax(p_mean)]
        epi = epistemic[torch.argmax(p_mean)]
    uncert = ale + epi
    return p_mean, uncert, ale, epi

def accuracy(inputs, target):
    _, max_indices = torch.max(inputs.data, 1)
    acc = (max_indices == target).sum().float() / max_indices.size(0)
    return acc.item()

def get_beta(batch_idx, m, beta_type, epoch, num_epochs, warmup_epochs=0):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    if epoch < warmup_epochs:
        beta /= warmup_epochs - epoch
    return beta

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        super(ThresholdPruning, self).__init__()
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold

class L1UnstructuredFFG(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, W_mu, W_rho, amount):
        super(L1UnstructuredFFG, self).__init__()
        self.amount = amount
        for _mu, _rho in zip(W_mu, W_rho):
            mu, rho = _mu[0].W_mu, _rho[0].W_rho
            snr = torch.abs(mu) / softplus(rho)
            snr_np = snr.detach().cpu().numpy()
            #idx = np.argpartion(snr_np)
            import pdb; pdb.set_trace()

    def compute_mask(self, mu, sigma, default_mask):
        snr = torch.abs(mu) / softplus(sigma)
        snr_np = snr.cpu().numpy()
        idx = np.argpartion(snr_np)
        import pdb; pdb.set_trace()

def prune_weights(net, mode='threshold', w_thresh=0., b_thresh=None, w_percentage=0., b_percentage=0.):
    weights_to_prune = [(m, 'weight') for m in net.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    biasses_to_prune = [(m, 'bias') for m in net.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]

    if mode == 'threshold':
        prune.global_unstructured(weights_to_prune, pruning_method=ThresholdPruning, threshold=w_thresh)
        prune.global_unstructured(biasses_to_prune, pruning_method=ThresholdPruning, threshold=b_thresh)
    elif mode == 'percentage':
        prune.global_unstructured(weights_to_prune, pruning_method=prune.L1Unstructured, amount=w_percentage)
        prune.global_unstructured(biasses_to_prune, pruning_method=prune.L1Unstructured, amount=b_percentage)

def prune_weights_mi(net, mode='threshold', w_thresh=0., b_thresh=None, w_percentage=0., b_percentage=0.):
    weights_to_prune = [(m, 'weight') for m in net.modules() if isinstance(m, (LinearRT, LinearLRT, Conv2dRT, Conv2dLRT))]
    biasses_to_prune = [(m, 'bias') for m in net.modules() if isinstance(m, (LinearRT, LinearLRT, Conv2dRT, Conv2dLRT))]

    if mode == 'threshold':
        prune.global_unstructured(weights_to_prune, pruning_method=ThresholdPruning, threshold=w_thresh)
        prune.global_unstructured(biasses_to_prune, pruning_method=ThresholdPruning, threshold=b_thresh)
    elif mode == 'percentage':
        prune.global_unstructured(weights_to_prune, pruning_method=prune.L1Unstructured, amount=w_percentage)
        prune.global_unstructured(biasses_to_prune, pruning_method=prune.L1Unstructured, amount=b_percentage)

def norm_grad(net):
    pass
