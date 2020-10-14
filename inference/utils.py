import torch
import torch.nn as nn
from torch.nn.utils import prune

import matplotlib.pyplot as plt
import numpy as np

def calc_uncert(img_list: torch.Tensor, reduction: str = 'mean'):
    img_list = torch.cat(img_list, dim=0)
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

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold

def prune_weights(net, mode='threshold', w_thresh=0., b_thresh=None, w_percentage=0., b_percentage=0.):
    weights_to_prune = [(m, 'weight') for m in net.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    biasses_to_prune = [(m, 'bias') for m in net.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]

    if mode == 'threshold':
        prune.global_unstructured(weights_to_prune, pruning_method=ThresholdPruning, threshold=w_thresh)
        prune.global_unstructured(biasses_to_prune, pruning_method=ThresholdPruning, threshold=b_thresh)
    elif mode == 'percentage':
        prune.global_unstructured(weights_to_prune, pruning_method=prune.L1Unstructured, amount=w_percentage)
        prune.global_unstructured(biasses_to_prune, pruning_method=prune.L1Unstructured, amount=b_percentage)

def norm_grad(net):
    pass
