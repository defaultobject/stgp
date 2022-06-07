import objax
from batchjax import batch_or_loop, BatchType

from . import Likelihood, Gaussian, BlockDiagonalGaussian
from ..utils.utils import ensure_module_list, can_batch, get_batch_type

def get_product_likelihood(likelihood_arr):
    if all(type(lik) == Gaussian for lik in likelihood_arr):
        return GaussianProductLikelihood(likelihood_arr)

    if all(type(lik) == BlockDiagonalGaussian for lik in likelihood_arr):
        return BlockGaussianProductLikelihood(likelihood_arr)

    return ProductLikelihood(likelihood_arr)


class ProductLikelihood(Likelihood):
    def __init__(self, likelihood_arr):
            super(ProductLikelihood, self).__init__()

            self.likelihood_arr = objax.ModuleList(likelihood_arr)


class GaussianProductLikelihood(ProductLikelihood):

    @property
    def variance(self):
        output_dim = len(self.likelihood_arr)
        var_arr = batch_or_loop(
            lambda lik:  lik.variance,
            [self.likelihood_arr],
            [0],
            dim = output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.likelihood_arr)
        )

        return var_arr


class BlockGaussianProductLikelihood(ProductLikelihood):
    pass

