import torch
from torch.nn.functional import softplus
import math

class MixtureNormal():
    def __init__(self, loc, scale, pi):
        assert(len(loc) == len(pi))
        assert(len(scale) == len(pi))
        if sum(pi) - 1 >= 1e-5:
            pi = pi / torch.sum(pi)

        self.loc = loc
        self.scale = scale
        self.pi = pi

    def log_prob(self, x):
        pdf = 0
        for i in range(len(self.pi)):
            pdf += self.pi[i].to(x.device) * torch.exp(Normal(self.loc[i], self.scale[i]).log_prob(x))
        return torch.log(pdf)

class Normal():
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def rsample(self):
        eps = torch.empty(self.loc.shape, dtype=self.loc.dtype, device=self.loc.device).normal_()
        return self.loc + eps * self.scale

    def log_prob(self, sample):
        var = (self.scale ** 2).to(sample.device)
        log_scale = self.scale.log().to(sample.device)
        return -((sample - self.loc.to(sample.device)) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
