from .module import ReparameterizationLayer, LocalReparameterizationLayer
from torch.nn.functional import conv2d

class Conv2DReparameterization(ReparameterizationLayer):

    def __init__(self,
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):

        super(Conv2DReparameterization, self).__init__(weight_posterior,
                                                       weight_prior,
                                                       bias_posterior,
                                                       bias_prior,
                                                       kl_divergence_fn,
                                                       conv2d,
                                                       stride=stride,
                                                       padding=padding,
                                                       dilation=dilation,
                                                       groups=groups)

class Conv2DLocalReparameterization(LocalReparameterizationLayer):

    def __init__(self,
                 weight_posterior,
                 weight_prior,
                 bias_posterior,
                 bias_prior,
                 kl_divergence_fn,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):

        super(Conv2DLocalReparameterization, self).__init__(weight_posterior,
                                                            weight_prior,
                                                            bias_posterior,
                                                            bias_prior,
                                                            kl_divergence_fn,
                                                            conv2d,
                                                            stride=stride,
                                                            padding=padding,
                                                            dilation=dilation,
                                                            groups=groups) 
