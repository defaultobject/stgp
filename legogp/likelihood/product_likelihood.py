
import objax

from . import Likelihood, Gaussian

def get_product_likelihood(likelihood_arr):
    if all(type(lik) == Gaussian for lik in likelihood_arr):
        return GaussianProductLikelihood(likelihood_arr)
    return ProductLikelihood(likelihood_arr)


class ProductLikelihood(Likelihood):
    def __init__(self, likelihood_arr):
            super(ProductLikelihood, self).__init__()

            self.likelihood_arr = objax.ModuleList(likelihood_arr)


class GaussianProductLikelihood(ProductLikelihood):
    pass

