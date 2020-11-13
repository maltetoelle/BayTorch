import torch
from torch.nn import Parameter, Module
from torch.nn.functional import softplus
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from ..distributions import mc_kl_divergence, MixtureNormal

class VIModule(Module):

    def __init__(self,
                 layer_fn,
                 weight_size,
                 bias_size=None,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse'):

        super(VIModule, self).__init__()

        self.layer_fn = layer_fn

        if prior is None:
            prior = {'mu': 0, 'sigma': 0.1}

        if posteriors is None:
            posteriors = {
                'mu': (0, 0.1),
                'rho': (-3., 0.1)
            }

        if 'pi' in list(prior.keys()):
            self._kl_divergence = mc_kl_divergence
            self.prior = MixtureNormal(prior['mu'], prior['sigma'], prior['pi'])
        else:
            self._kl_divergence = kl_divergence
            self.prior = Normal(prior['mu'], prior['sigma'])

        self.kl_type = kl_type

        self.posterior_mu_initial = posteriors['mu']
        self.posterior_rho_initial = posteriors['rho']

        # self.weight = Parameter(torch.empty(2, *weight_size))
        # if bias_size is not None:
        #     self.bias = Parameter(torch.empty(2, bias_size))
        # else:
        #     self.register_parameter('bias', None)

        self.W_mu = Parameter(torch.empty(weight_size))
        self.W_rho = Parameter(torch.empty(weight_size))
        if bias_size is not None:
            self.bias_mu = Parameter(torch.empty(bias_size))
            self.bias_rho = Parameter(torch.empty(bias_size))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data[0].normal_(*self.posterior_mu_initial)
        # self.weight.data[1].normal_(*self.posterior_rho_initial)
        # if self.bias is not None:
        #     self.bias.data[0].normal_(*self.posterior_mu_initial)
        #     self.bias.data[1].normal_(*self.posterior_rho_initial)

        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias_mu is not None:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    @property
    def _kl(self):
        kl = self.kl_divergence(Normal(self.W_mu.cpu(), softplus(self.W_rho).cpu()), self.prior, self.kl_type).sum()
        if self.bias_mu is not None:
            kl += self.kl_divergence(Normal(self.bias_mu.cpu(), softplus(self.bias_rho).cpu()), self.prior, self.kl_type).sum()
        return kl

    # @property
    # def _kl(self):
    #     kl = self.kl_divergence(Normal(self.weight[0].cpu(), softplus(self.weight[1]).cpu()), self.prior, self.kl_type).sum()
    #     if self.bias is not None:
    #         kl += self.kl_divergence(Normal(self.bias[0].cpu(), softplus(self.bias[1]).cpu()), self.prior, self.kl_type).sum()
    #     return kl

    def kl_divergence(self, p, q, kl_type='reverse'):
        if kl_type == 'reverse':
            return self._kl_divergence(q, p)
        else:
            return self._kl_divergence(p, q)

    @staticmethod
    def rsample(mu, sigma):
        eps = torch.empty(mu.size()).normal_(0, 1).to(mu.device)
        return mu + eps * sigma
