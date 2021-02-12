from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity

from ..approximate_posteriors import *

from ..inference import *
from ..computation import *

from ..computation.marginal_likelihoods import gaussian_gaussian_marginal_likelihood
from ..computation.predictors import gp_predict_y, gp_predict_y_diagional
from ..decorators import return_gradients, set_defaults_from_self, ensure_data_passed
from ..distributions import *


from ..data import Data, ListData

from .. import Dispatcher

import jax.numpy as np
from jax import jit, partial

import typing 
from typing import Union, List, Optional


@Dispatcher.register('gp_model', BatchInference, None, None)
class BatchGP(Model):
    def setup(self):
        self.data = ListData(self.X, self.Y, self.X_onp, self.Y_onp)
        #only setup if required
        #if not self.set_defaults: return 

        self.prior_arr = []
        for p in range(self.num_outputs):
            prior = KernelGaussianDistribution(
                kernel=self.kernel_arr[p],
                meta={
                    'X': self.X
                }   
            ) 
            self.prior_arr.append(prior)

    @return_gradients
    @set_defaults_from_self
    def get_objective(self, data:Optional[Data] = None):

        Y_arr = data.Y
        X_arr = data.X
        ml = 0.0

        ml_fn = Dispatcher.dispatch('marginal_likelihood', type(self.likelihood),  type(self.prior_arr[0]))

        #assumes num_outputs = 1
        for p in range(self.num_outputs):
            ml_p = ml_fn(self.X_arr[p], Y_arr[p], self.likelihood, self.prior_arr[p])
            ml += ml_p

        return -ml

    @set_defaults_from_self
    def posterior(self, data: Optional[Data]=None, diagonal_var: Optional[bool] = True):
        X = data.X[0]
        return self.predict_y(X, data, diagonal_var=diagonal_var)
                
    @ensure_data_passed
    @set_defaults_from_self
    def predict_y(self, data_xs:Data, data:Optional[Data] = None, diagonal_var: Optional[bool]=True):
        Y = data.Y
        X = data.X

        #data_xs is a placeholder data so do not return a list
        XS = data_xs.X

        #get function to compute prediction
        fun = Dispatcher.dispatch('predictors', type(self.likelihood),  diagonal=diagonal_var)

        #TODO: hard coded for now
        p = 0

        mu_arr, sig_arr = [], []
        mu, sig = fun(XS, X[p], Y[p], self.likelihood, self.prior_arr[p])
        print(self.prior_arr[p].kernel.variance)

        #ensure correct sizes
        if diagonal_var:
            mu = np.reshape(mu, [mu.shape[0], 1])
            sig = np.reshape(sig, [sig.shape[0], 1])

        mu_arr.append(mu)
        sig_arr.append(sig)

        return mu_arr, sig_arr
        
    @partial(jit, static_argnums=(2))
    def predict_f(self, XS:np.ndarray, diagonal_var:bool):
        raise NotImplementedError('Predict f is not implemented yet')

