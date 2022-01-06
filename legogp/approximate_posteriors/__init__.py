from .approximate_posterior import ApproximatePosterior
from .gaussian_approximate_posterior import GaussianApproximatePosterior
from .mean_field_approximate_posterior import MeanFieldApproximatePosterior
from .mm_gaussian_inner_layer_posterior import MM_GaussianInnerLayerApproximatePosterior

__all__ = [
    'ApproximatePosterior', 
    'GaussianApproximatePosterior',
    'MeanFieldApproximatePosterior',
    'MM_GaussianInnerLayerApproximatePosterior'
]
