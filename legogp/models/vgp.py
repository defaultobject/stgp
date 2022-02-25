import objax
import jax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple
import chex

from .. import settings
from ..decorators import strict_mode_check, ensure_data
from ..dispatch import dispatch
from ..dispatch import evoke

from ..core import Model, Posterior
from . import GP, BatchGP
from ..kernels import Kernel
from ..inference import Variational
from ..transforms import Transform, Identity
from ..computation.log_marginal_likelihoods import *
from ..likelihood import get_product_likelihood
from ..kernels import RBF
from ..approximate_posteriors import MeanFieldApproximatePosterior
from ..sparsity import NoSparsity
from ..utils.utils import ensure_module_list
from ..defaults import get_default_kernel, get_default_likelihood, get_default_independent_prior
from ..computation.natural_gradients.nat_grad import general_ell_natural_gradients


@dispatch(Model, 'Variational')
class VGP(Posterior):
    def __init__(
        self, 
        X=None, 
        Y=None, 
        Z = None,
        inference: 'Variational'=None, 
        approximate_posterior: 'Posterior'=None, 
        likelihood: 'Likelihood'=None, 
        kernel: 'Kernel'=None, 
        prior: 'Transform'=None, 
        whiten=False, 
        minibatch_size=None,
        **kwargs
    ):

        # will save X, Y as a property
        super(VGP, self).__init__(X, Y, **kwargs)

        self.inference = inference
        self._likelihood = likelihood
        self._prior = prior
        self.kernel = kernel
        self.approximate_posterior = approximate_posterior
        self.whiten = whiten
        self.minibatch_size = minibatch_size
        self._Z = Z  

        self.set_defaults()
        self.fix_inputs()

    def log_marginal_likelihood(self, X=None, Y=None):
        raise NotImplementedError()

    @property
    def output_dim(self): return self.Y.shape[1]

    @property
    def input_dim(self): return self.output_dim

    @property
    def input_space_dim(self): return self.X.shape[1]

    @property
    def likelihood(self): return self._likelihood

    @property
    def prior(self): return self._prior

    def setup_data(self):
        """ Ensure input data is valid """

        if self.X is None:
            raise RuntimeError('X must be passed')

        if self.Y is None:
            raise RuntimeError('Y must be passed')

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def fix_inputs(self):
        """ Convert all inputs into a consistent format """

        if type(self.likelihood) == list:
            self._likelihood = get_product_likelihood(self._likelihood)

    def set_defaults(self):
        # Figure out which prior mode is being used (kernel vs prior)

        if (self.kernel is not None) and (self.prior is not None):
            raise RuntimeError('Only kernel or a prior must be passed')

        if self.prior is None:
            # construct an independent prior for each latent function

            # Only set a default kernel if we are in kernel mode and one has not been passed
            if self.kernel is None:
                self.kernel = get_default_kernel(self.input_space_dim, self.input_dim)
            else:
                if type(self.kernel) is not list:
                    self.kernel = [self.kernel]

            # Construct independent prior

            self._prior = get_default_independent_prior(
                self.X,
                self.input_space_dim, 
                self.input_dim, 
                kernel_list=self.kernel,
                Z = self._Z,
            )

        if self.inference == None:
            self.inference = Variational(whiten=self.whiten, minibatch_size=self.minibatch_size)

        if self.likelihood == None:
            # Default Gaussian liklelihood
            self._likelihood = get_default_likelihood(self.output_dim)

        if self.approximate_posterior is None:
            self.approximate_posterior = MeanFieldApproximatePosterior(
                dim_list=[self.Y.shape[0]]*self.prior.num_latents
            )

    def get_objective(self):

        elbo = self.inference.ELBO(
            self.X,
            self.Y,
            self.likelihood,
            self.prior,
            self.approximate_posterior
        )

        return -elbo

    def mean(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False)

        chex.assert_shape(mu, [self.output_dim, XS.shape[0], 1])
        return mu

    def var(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)

        chex.assert_shape(var, [self.output_dim, XS.shape[0], 1])
        return var

    def covar(self, XS_1, XS_2, X=None, Y=None):
        var_arr =  self.inference.predictive_covar(
            XS_1, 
            XS_2, 
            self.X, 
            self.Y, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior
        )

        chex.assert_shape(var_arr, [self.output_dim, XS_1.shape[0], XS_2.shape[0]])
        return var_arr

    def predict_latents(self, XS, diagonal=True, squeeze=True):
        mean, var = self.inference.predict_latents(
            XS, 
            self.X, 
            self.Y, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var

    def predict_f(self, XS, diagonal=True, squeeze=True):

        mean, var = self.inference.predict_f(
            XS, 
            self.X, 
            self.Y, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var

    def predict_y(self, XS, diagonal=True, squeeze=True):
        X, Y = self.X, self.Y

        mu_arr, var_arr =  self.inference.predict_y(
            XS, 
            self.X, 
            self.Y, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr) 

        return mu_arr, var_arr

    def natural_gradients(self, learning_rate, m_var, s_chol_var):
        return general_ell_natural_gradients(
            self,
            learning_rate,
            [m_var, s_chol_var]
        )
