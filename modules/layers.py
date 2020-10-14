import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

from .module import RTLayer, LRTLayer
from .utils import mean_field_normal_initializer, default_prior, mc_kl_divergence, scale_mixture_prior #, kl_divergence
from torch.nn.modules.utils import _pair

class LinearRT(RTLayer):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 posterior={"loc": 0.0, "ro": -3.0},
                 prior_scale=1,
                 prior_pi=None):

        if prior_pi is not None:
            prior = scale_mixture_prior(prior_scale, prior_pi)
            kl_divergence_fn = mc_kl_divergence
        else:
            prior = default_prior(prior_scale)
            kl_divergence_fn = kl_divergence

        if bias:
            bias_posterior = mean_field_normal_initializer(out_features, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(LinearRT, self).__init__(
            weight_posterior=mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"]),
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.linear)

class Conv2dRT(RTLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 posterior={"loc": 0.0, "ro": -3.0},
                 prior_scale=1,
                 prior_pi=None):

        kernel_size = _pair(kernel_size)

        if prior_pi is not None:
            prior = scale_mixture_prior(prior_scale, prior_pi)
            kl_divergence_fn = mc_kl_divergence
        else:
            prior = default_prior(prior_scale)
            kl_divergence_fn = kl_divergence

        if bias:
            bias_posterior = mean_field_normal_initializer(out_channels, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(Conv2dRT, self).__init__(
            weight_posterior=mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"]),
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.Conv2d,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups)

class LinearLRT(LRTLayer):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 posterior={"loc": 0.0, "ro": -3.0},
                 prior_scale=1,
                 prior_pi=None):

        if prior_pi is not None:
            prior = scale_mixture_prior(prior_scale, prior_pi)
            kl_divergence_fn = mc_kl_divergence
        else:
            prior = default_prior(prior_scale)
            kl_divergence_fn = kl_divergence

        if bias:
            bias_posterior = mean_field_normal_initializer(out_features, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(LinearLRT, self).__init__(
            weight_posterior=mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"]),
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.linear)


class Conv2dLRT(LRTLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 posterior={"loc": 0.0, "ro": -3.0},
                 prior_scale=1,
                 prior_pi=None):

        kernel_size = _pair(kernel_size)
        if prior_pi is not None:
            prior = scale_mixture_prior(prior_scale, prior_pi)
            kl_divergence_fn = mc_kl_divergence
        else:
            prior = default_prior(prior_scale)
            kl_divergence_fn = kl_divergence

        if bias:
            bias_posterior = mean_field_normal_initializer(out_channels, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(Conv2dLRT, self).__init__(
            weight_posterior=mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"]),
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.Conv2d,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups)
