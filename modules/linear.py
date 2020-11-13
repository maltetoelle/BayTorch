from torch.nn.functional import linear

from .reparam_layers import RTLayer, LRTLayer

class LinearRT(RTLayer):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse',
                 _version='old'):

        self.in_features = in_features
        self.out_featurs = out_features

        weight_size = (out_features, in_features)

        bias_size = (out_features) if bias else None

        super(LinearRT, self).__init__(layer_fn=linear,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        _version=_version)

class LinearLRT(LRTLayer):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior=None,
                 posteriors=None,
                 kl_type='reverse',
                 _version='old'):

        self.in_features = in_features
        self.out_featurs = out_features

        weight_size = (out_features, in_features)

        bias_size = (out_features) if bias else None

        super(LinearLRT, self).__init__(layer_fn=linear,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        _version=_version)
