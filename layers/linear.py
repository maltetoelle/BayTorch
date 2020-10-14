import torch.nn.functional as F

from .module import ReparameterizationLayer, LocalReparameterizationLayer
from .utils import mean_field_normal_initializer, default_prior, kl_divergence, mc_kl_divergence, scale_mixture_prior

class LinearRT(ReparameterizationLayer):

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

        weight_posterior = mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"])

        if bias:
            bias_posterior = mean_field_normal_initializer(out_features, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(LinearRT, self).__init__(
            weight_posterior=weight_posterior,
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.linear)

class LinearLRT(LocalReparameterizationLayer):

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

        weight_posterior = mean_field_normal_initializer((out_features, in_features), posterior["loc"], posterior["ro"])

        if bias:
            bias_posterior = mean_field_normal_initializer(out_features, posterior["loc"], posterior["ro"])
            bias_prior = prior
        else:
            bias_posterior, bias_prior = None, None

        super(LinearLRT, self).__init__(
            weight_posterior=weight_posterior,
            weight_prior=prior,
            bias_posterior=bias_posterior,
            bias_prior=bias_prior,
            kl_divergence_fn=kl_divergence_fn,
            layer_fn=F.linear)
