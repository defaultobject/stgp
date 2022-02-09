import objax
import jax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple
import chex

from .. import settings
from ..decorators import strict_mode_check, ensure_data
from ..dispatch import dispatch
from ..batching import loop_or_batch

from ..core import Model, Posterior
from . import GP
from ..kernels import Kernel
from ..inference import Batch
from ..likelihood import Gaussian, ProductLikelihood
from ..kernels import RBF
from ..utils.utils import ensure_module_list
from ..transforms import Independent
from ..defaults import get_default_kernel, get_default_likelihood, get_default_independent_prior

from ..sparsity import NoSparsity

import warnings

@dispatch('Model', 'Batch')
class BatchGP(Posterior):
    def __init__(self, X=None, Y=None, inference: 'Batch'=None, likelihood: 'Likelihood'=None, kernel: 'Kernel'=None, prior: 'Transform' = None, **kwargs):

        if X is None:
            raise RuntimeError('X must be passed')

        if Y is None:
            raise RuntimeError('Y must be passed')

        # will save X, Y as a property
        super(BatchGP, self).__init__(X, Y, **kwargs)

        self.inference = inference
        self._likelihood = likelihood
        self.kernel = kernel
        self._prior = prior
        self.sparsity = NoSparsity(self.X) # Sparsity for batch GPs is not supported

        self.set_defaults()

    @property
    def likelihood(self): return self._likelihood

    @property
    def prior(self): return self._prior

    @property
    def input_space_dim(self): return self.X.shape[1]

    @property
    def output_dim(self): return self.Y.shape[1]

    @property
    def input_dim(self): return self.output_dim

    def set_defaults(self):
        """ Replace missing options with defaults """

        # Figure out which prior mode is being used (kernel vs prior)

        if (self.kernel is not None) and (self.prior is not None):
            raise RuntimeError('Only kernel or a prior must be passed')

        if self.prior is None:
            # construct an independent prior for each latent function

            # Only set a default kernel if we are in kernel mode and one has not been passed
            if self.kernel is None:
                self.kernel = get_default_kernel(self.input_space_dim, self.output_dim)
            else:
                if type(self.kernel) is not list:
                    self.kernel = [self.kernel]

            # Construct independent prior

            self._prior = get_default_independent_prior(
                self.X,
                self.input_space_dim, 
                self.output_dim, 
                kernel_list=self.kernel
            )

        if self.inference == None:
            self.inference = Batch()

        if self.likelihood == None:
            self._likelihood = get_default_likelihood(self.output_dim)

    def log_marginal_likelihood(self, X=None, Y=None):

        if X is None:
            X, Y = self.X, self.Y

        nlml = self.inference.neg_log_marginal_likelihood(
            X,
            Y,
            self, 
            self.likelihood,
            self.prior
        )

        chex.assert_rank(nlml, 0)

        return nlml

    def get_objective(self, X=None, Y=None):
        return self.log_marginal_likelihood(X, Y)

    def mean(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False)
        return mu

    def var(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)
        return var

    def covar(self, XS_1, XS_2, X=None, Y=None):
        if X is None:
            X, Y = self.X, self.Y

        var_arr =  self.inference.predictive_covar(
            XS_1, XS_2, X, Y, self, self.likelihood, self.prior
        )
        return var_arr

    def predict_f(self, XS,  X=None, Y=None, diagonal=True, squeeze=True):
        if X is None:
            X, Y = self.X, self.Y

        mu_arr, var_arr =  self.inference.predict_f(
            XS, X, Y, self, self.likelihood, self.prior, diagonal=diagonal
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr) 

        return mu_arr, var_arr

    def predict_y(self, XS, diagonal=True, squeeze=True):
        X, Y = self.X, self.Y

        mu_arr, var_arr =  self.inference.predict_y(
            XS, X, Y, self, self.likelihood, self.prior, diagonal=diagonal
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr) 

        return mu_arr, var_arr
