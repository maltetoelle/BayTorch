import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.layers import *

class MeanFieldVI(nn.Module):

    def __init__(self, net, prior_scale=1., prior_pi=None, reparam='local'):
        super(MeanFieldVI, self).__init__()
        self.net = net

        if reparam == 'local':
            # self._conv2d = BBBConv2d
            self._conv2d = Conv2dLRT
            self._linear = LinearLRT
        else:
            self._conv2d = Conv2dRT
            self._linear = LinearRT

        self._replace_deterministic_modules(self.net, prior_scale, prior_pi)
        self.net.kl = self.kl

    def forward(self, x):
        return self.net(x)

    @property
    def kl(self):
        kl = 0
        for layer in self.modules():
            if hasattr(layer, '_kl'):
                kl += layer._kl
        return kl

    def _replace_deterministic_modules(self, module, prior_scale, prior_pi):
        for key, _module in module._modules.items():
            if len(_module._modules):
                self._replace_deterministic_modules(_module, prior_scale, prior_pi)
            else:
                if isinstance(_module, nn.Linear):
                    layer = self._linear(_module.in_features, _module.out_features, torch.is_tensor(_module.bias), prior_scale=prior_scale, prior_pi=prior_pi)
                    module._modules[key] = layer
                elif isinstance(_module, nn.Conv2d):
                    layer = self._conv2d(
                        in_channels=_module.in_channels,
                        out_channels=_module.out_channels,
                        kernel_size=_module.kernel_size,
                        stride=_module.stride,
                        padding=_module.padding,
                        dilation=_module.dilation,
                        bias=torch.is_tensor(_module.bias))#, groups=_module.groups, prior_scale=prior_scale, prior_pi=prior_pi)
                    module._modules[key] = layer

class MCDropoutVI(nn.Module):

    def __init__(self, net, dropout_type='1d',
                 dropout_p=0.5, deterministic_output=False,
                 output_dip_drop=False):

        super(MCDropoutVI, self).__init__()
        self.net = net
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p

        self._replace_deterministic_modules(self.net)
        # self.deterministic_output = deterministic_output
        if deterministic_output:
            self._make_last_layer_deterministic(self.net)
        if not output_dip_drop:
            self._dip_make_output_deterministic(self.net)

    def forward(self, x):
        return self.net(x)

    def _replace_deterministic_modules(self, module):
        for key, _module in module._modules.items():
            if len(_module._modules):
                self._replace_deterministic_modules(_module)
            else:
                if isinstance(_module, (nn.Linear, nn.Conv2d)):
                    module._modules[key] =  MCDropout(_module, self.dropout_type, self.dropout_p)

    def _make_last_layer_deterministic(self, module):
        for i, (key, layer) in enumerate(module._modules.items()):
            if i == len(module._modules) - 1:
                if isinstance(layer, MCDropout):
                    module._modules[key] = layer.layer
                elif len(layer._modules):
                    self._make_last_layer_deterministic(layer)

    def _dip_make_output_deterministic(self, module):
        for i, (key, layer) in enumerate(module._modules.items()):
            if type(layer) == nn.Sequential:
                for name, m in layer._modules.items():
                    if type(m) == MCDropout:
                        layer._modules[name] = m.layer
