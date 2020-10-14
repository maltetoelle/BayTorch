import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from .distributions import Normal

class VIModule(nn.Module):

    def __init__(self,
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn):

        super(VIModule, self).__init__()

        self.weight_loc = weight_posterior['loc']
        self.weight_ro = weight_posterior['ro']
        self.weight_prior = weight_prior

        if bias_posterior is None:
            self.bias_loc = None
            self.bias_ro = None
            self.bias_prior = None
        else:
            self.bias_loc = bias_posterior['loc']
            self.bias_ro = bias_posterior['ro']
            self.bias_prior = bias_prior

        self.kl_divergence = kl_divergence_fn

    @property
    def _kl(self):
        kl = torch.sum(self.kl_divergence(Normal(self.weight_loc, softplus(self.weight_ro)), self.weight_prior))
        if self.bias_loc is not None:
            kl += torch.sum(self.kl_divergence(Normal(self.bias_loc, softplus(self.bias_ro)), self.bias_prior))
        return kl

    @staticmethod
    def rsample(loc, scale):
        eps = torch.empty(loc.shape, dtype=loc.dtype, device=loc.device).normal_()
        return loc + eps * scale

    def extra_repr(self):
        s = "weight: {}, bias: {}".format(list(self.weight_loc.size()), list(self.bias_loc.size()))
        return s

class ReparameterizationLayer(VIModule):

    def __init__(self,
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 layer_fn,
                 **kwargs):

        super(ReparameterizationLayer, self).__init__(weight_posterior,
                                                      weight_prior,
                                                      bias_posterior,
                                                      bias_prior,
                                                      kl_divergence_fn)
        self.layer_fn = layer_fn
        self.kwargs = kwargs

    def forward(self, input):
        if self.bias_loc is None:
            return self.layer_fn(input, self.rsample(self.weight_loc, softplus(self.weight_ro)), **self.kwargs)
        return self.layer_fn(input, self.rsample(self.weight_loc, softplus(self.weight_ro)), self.rsample(self.bias_loc, softplus(self.bias_ro)), **self.kwargs)

class LocalReparameterizationLayer(VIModule):

    def __init__(self,
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 layer_fn,
                 **kwargs):

        super(LocalReparameterizationLayer, self).__init__(weight_posterior,
                                                           weight_prior,
                                                           bias_posterior,
                                                           bias_prior,
                                                           kl_divergence_fn)
        self.layer_fn = layer_fn
        self.kwargs = kwargs

    def forward(self, input):
        if self.bias_loc is None:
            output_loc = self.layer_fn(input, self.weight_loc, **self.kwargs)
            output_scale = torch.sqrt(1e-9 + self.layer_fn(input.pow(2), softplus(self.weight_ro)**2, **self.kwargs))
            return self.rsample(output_loc, output_scale)
        output_loc = self.layer_fn(input, self.weight_loc, self.bias_loc, **self.kwargs)
        output_scale = torch.sqrt(1e-9 + self.layer_fn(input.pow(2), softplus(self.weight_ro)**2, softplus(self.bias_ro)**2, **self.kwargs))
        return self.rsample(output_loc, output_scale)
