from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter

from .module import ReparameterizationLayer, LocalReparameterizationLayer
from .utils import mean_field_normal_initializer, default_prior, kl_divergence, mc_kl_divergence, scale_mixture_prior, normal_initializer

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class BBBConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.conv2d(
            x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        act_var = 1e-16 + F.conv2d(
            x ** 2, self.W_sigma ** 2, bias_var, self.stride, self.padding, self.dilation, self.groups)
        act_std = torch.sqrt(act_var)

        # if self.training or sample:
        eps = torch.empty(act_mu.size()).normal_(0, 1).to(act_mu.device)
        return act_mu + act_std * eps
        # else:
        #     return act_mu

    @property
    def _kl(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


# class Conv2dLRT(nn.Module):
#
#     def __init__(self,
#                 in_channels: int,
#                 out_channels: int,
#                 kernel_size: int,
#                 stride: int = 1,
#                 padding: int = 0,
#                 dilation: int = 1,
#                 bias: bool = True,
#                 groups: int = 1,
#                 posterior: dict = {'mu': 0., 'rho': -3.}
#                 prior_scale: float = 1.,
#                 prior_pi: float = None):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.use_bias = bias
#
#         self.W_mu = normal_initializer((out_channels, in_channels // groups, kernel_size, kernel_size), posterior['mu'], 0.03)
#         self.W_rho = normal_initializer((out_channels, in_channels // groups, kernel_size, kernel_size), posterior['rho'], 0.03)
#
#         if self.use_bias:
#             self.bias_mu = normal_initializer(out_channels, posterior['mu'], 0.03)
#             self.bias_rho = normal_initializer(out_channels, posterior['rho'], 0.03)
#         else:
#             self.register_parameter('bias_mu', None)
#             self.register_parameter('bias_rho', None)
#
#         if prior_pi is not None:
#             prior = scale_mixture_prior(prior_scale, prior_pi)
#             self.kl_divergence = mc_kl_divergence
#         else:
#             prior = default_prior(prior_scale)
#             self.kl_divergence = kl_divergence
#
#     @property
#     def _kl(self):
#         kl = self.kl_divergence()
#         kl = torch.sum(self.kl_divergence(Normal(self.weight_loc, softplus(self.weight_ro)), self.weight_prior))
#         if self.bias_loc is not None:
#             kl += torch.sum(self.kl_divergence(Normal(self.bias_loc, softplus(self.bias_ro)), self.bias_prior))
#         return kl
#
#     def forward(self, x):
#         return x

class Conv2dRT(ReparameterizationLayer):#bnn.Conv2dRT):

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

        weight_posterior = mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"])

        if bias:
            bias_posterior = mean_field_normal_initializer(out_channels, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(Conv2dRT, self).__init__(
            weight_posterior=weight_posterior,
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.conv2d,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups)


class Conv2dLRT(LocalReparameterizationLayer):

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

        weight_posterior = mean_field_normal_initializer((out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), posterior["loc"], posterior["ro"])

        if bias:
            bias_posterior = mean_field_normal_initializer(out_channels, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(Conv2dLRT, self).__init__(
            weight_posterior=weight_posterior,
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.conv2d,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups)
