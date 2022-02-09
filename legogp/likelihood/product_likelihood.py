
import objax

from . import Likelihood

class ProductLikelihood(Likelihood):
    def __init__(self, likelihood_arr):
            super(ProductLikelihood, self).__init__()

            self.likelihood_arr = objax.ModuleList(likelihood_arr)
