from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter

class VIModule(nn.Module):

    def __init__(self, bias=True, priors=None, device=1):
        super(VIModule, self).__init__()

        self.use_bias = bias
        self.device = torch.device("cuda:%d" % device if torch.cuda.is_available() else "cpu")

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

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    @property
    def _kl(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

    @staticmethod
    def calculate_kl(mu_q, sig_q, mu_p, sig_p):
        kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
        return kl

class LRTLayer(VIModule):

    def __init__(self, layer_fn, bias=True, priors=None, device=1, **kwargs):
        super(LRTLayer, self).__init__(bias, priors, device)
        self.layer_fn = layer_fn
        self.kwargs = kwargs

    def forward(self, x, sample=True):
        self.W_sigma = F.softplus(self.W_rho)
        if self.use_bias:
            self.bias_sigma = F.softplus(self.bias_rho)
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = self.layer_fn(x, self.W_mu, self.bias_mu, **self.kwargs)
        act_std = torch.sqrt(1e-16 + self.layer_fn(x**2, self.W_sigma**2, bias_var, **self.kwargs))

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(act_mu.device)
            return act_mu + act_std * eps
        else:
            return act_mu

class Conv2dLRT(LRTLayer):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1,
                 padding=0, dilation=1, priors=None, device=1):

        super(Conv2dLRT, self).__init__(bias, priors, device, stride, padding, dilation, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()
