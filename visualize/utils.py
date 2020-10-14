import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import Conv2dRT, Conv2dLRT, LinearRT, LinearLRT

def get_params(net):
    _net = copy.deepcopy(net)
    params = [torch.flatten(p.requires_grad_(False)) for p in _net.parameters() if p.requires_grad]
    params = torch.cat(params).cpu().numpy()
    return params

def get_params_mi(net):
    _net = nn.Sequential(net._modules)
    _net.load_state_dict(net.state_dict())

    mus = []
    sigmas = []
    for module in _net.modules():
        if isinstance(module, (Conv2dRT, Conv2dLRT, LinearRT, LinearLRT)):
            mus.append(torch.flatten(module.weight_loc.requires_grad_(False)))
            sigmas.append(torch.flatten(F.softplus(module.weight_ro.requires_grad_(False))))
    return torch.cat(mus).cpu().numpy(), torch.cat(sigmas).cpu().numpy()
