import torch.nn.functional as F

from .module import ReparameterizationLayer, LocalReparameterizationLayer
from .utils import mean_field_normal_initializer, default_prior, kl_divergence, mc_kl_divergence, scale_mixture_prior
from torch.nn.modules.utils import _pair

class LinearReparameterization(ReparameterizationLayer):

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
            super(LinearReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=mean_field_normal_initializer(out_features, posterior["loc"], posterior["ro"]),
                bias_prior=prior,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.linear)
        else:
            super(LinearReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=None,
                bias_prior=None,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.linear)

class Conv2DReparameterization(ReparameterizationLayer):

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
            super(Conv2DReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=mean_field_normal_initializer(out_channels, posterior["loc"], posterior["ro"]),
                bias_prior=prior,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.conv2d,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups)
        else:
            super(Conv2DReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=None,
                bias_prior=None,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.conv2d,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups)

class LinearLocalReparameterization(LocalReparameterizationLayer):

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
            super(LinearLocalReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=mean_field_normal_initializer(out_features, posterior["loc"], posterior["ro"]),
                bias_prior=prior,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.linear)
        else:
            super(LinearLocalReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=None,
                bias_prior=None,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.linear)

class Conv2DLocalReparameterization(LocalReparameterizationLayer):

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
            super(Conv2DLocalReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=mean_field_normal_initializer(out_channels, posterior["loc"], posterior["ro"]),
                bias_prior=prior,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.conv2d,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups)
        else:
            super(Conv2DLocalReparameterization, self).__init__(
                weight_posterior=mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"]),
                weight_prior=prior,
                bias_posterior=None,
                bias_prior=None,
                kl_divergence_fn=kl_divergence_fn,
                layer_fn=F.conv2d,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups)
