import torch
import numpy as np
import sys
import time
from torch.nn import Parameter
from torch.distributions.normal import Normal
from .distributions import MixtureNormal

def normal_initializer(size, mean=0.0, std=0.1):
    return Parameter(torch.normal(mean=mean*torch.ones(size), std=std))

def mean_field_normal_initializer(size, loc=0.0, ro=-3.0):
    return {"loc": normal_initializer(size, mean=loc), "ro": normal_initializer(size, mean=ro)}

def default_prior(scale=1., size=1):
    return Normal(loc=torch.zeros(size), scale=scale*torch.ones(size))

def scale_mixture_prior(scale=[10, 0.01], pi=[.5, .5]):
    return MixtureNormal(loc=Parameter(torch.zeros(len(scale)), requires_grad=False),
                         scale=Parameter(torch.tensor(scale), requires_grad=False),
                         pi=Parameter(torch.tensor(pi), requires_grad=False))

def mc_kl_divergence(p, q, n_samples=1):
    kl = 0
    for _ in range(n_samples):
        sample = p.rsample()
        kl += p.log_prob(sample) - q.log_prob(sample)
    return kl / n_samples
