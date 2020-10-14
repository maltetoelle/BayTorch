from .module import ReparameterizationLayer, LocalReparameterizationLayer
from torch.nn.functional import linear

class LinearReparameterization(ReparameterizationLayer):

    def __init__(self,
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn):

        super(LinearReparameterization, self).__init__(weight_posterior,
                                                       weight_prior,
                                                       bias_posterior,
                                                       bias_prior,
                                                       kl_divergence_fn,
                                                       linear)

class LinearLocalReparameterization(LocalReparameterizationLayer):

    def __init__(self,
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn):

        super(LinearLocalReparameterization, self).__init__(weight_posterior,
                                                            weight_prior,
                                                            bias_posterior,
                                                            bias_prior,
                                                            kl_divergence_fn,
                                                            linear)
