import torch
from torch.nn import Parameter
from .distributions import Normal, MixtureNormal

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
    # return MixtureNormal(loc=torch.zeros(len(scale)),
    #                      scale=torch.tensor(scale),
    #                      pi=torch.tensor(pi))

def mc_kl_divergence(p, q, n_samples=1):
    kl = 0
    for _ in range(n_samples):
        sample = p.rsample()
        kl += p.log_prob(sample) - q.log_prob(sample)
    return kl / n_samples

def kl_divergence(q, p):
    q_loc, q_scale = q.loc.to(p.loc.device), q.scale.to(p.loc.device)
    kl = 0.5 * (2 * torch.log(p.scale / q_scale) - 1 + (q_scale / p.scale).pow(2) + ((p.loc - q_loc) / p.scale).pow(2)).sum()
    return kl
# def kl_divergence(p, q):
#     q_loc, q_scale = q.loc.to(p.loc.device), q.scale.to(p.loc.device)
#     kl = ((p.scale / q_scale)**2).log() + (q_scale**2 + (q_loc - p.loc)**2) / (2 * p.loc**2) - 0.5

    # var_ratio = (p.scale / q.scale.to(p.loc.device)).pow(2)
    # t1 = ((p.loc - q.loc.to(p.loc.device)) / q.scale.to(p.loc.device)).pow(2)
    # kl2 = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

    # import pdb; pdb.set_trace()
    # return kl

# def kl_divergence(p, q):
#     var_ratio = (p.scale / q.scale.to(p.loc.device)).pow(2)
#     t1 = ((p.loc - q.loc.to(p.loc.device)) / q.scale.to(p.loc.device)).pow(2)
#     return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
