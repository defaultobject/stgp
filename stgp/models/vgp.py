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
#from ..computation.natural_gradients.nat_grad import general_ell_natural_gradients


@dispatch(Model, 'Variational')
class VGP(Posterior):
    def __init__(
        self, 
        X=None, 
        Y=None, 
        Z = None,
        data = None,
        inference: 'Variational'=None, 
        approximate_posterior: 'Posterior'=None, 
        likelihood: 'Likelihood'=None, 
        kernel: 'Kernel'=None, 
        prior: 'Transform'=None, 
        whiten=False, 
        minibatch_size=None,
        ell_samples=None,
        prediction_samples=None,
        **kwargs
    ):

        # will save X, Y as a property
        super(VGP, self).__init__(X, Y, data, **kwargs)

        self.inference = inference
        self._likelihood = likelihood
        self._prior = prior
        self.kernel = kernel
        self.approximate_posterior = approximate_posterior
        self.whiten = whiten
        self.minibatch_size = minibatch_size
        self._Z = Z  
        self.ell_samples = ell_samples
        self.prediction_samples = prediction_samples

        self.set_defaults()
        self.fix_inputs()

    def log_marginal_likelihood(self, X=None, Y=None):
        raise NotImplementedError()

    @property
    def output_dim(self): return self.data.Y.shape[1]

    @property
    def input_dim(self): return self.output_dim

    @property
    def input_space_dim(self): return self.data.X.shape[1]

    @property
    def likelihood(self): return self._likelihood

    @property
    def prior(self): return self._prior

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
            self.inference = Variational(
                whiten=self.whiten,
                minibatch_size=self.minibatch_size,
                ell_samples = self.ell_samples,
                prediction_samples = self.prediction_samples
            )

        if self.likelihood == None:
            # Default Gaussian liklelihood
            self._likelihood = get_default_likelihood(self.output_dim)

        if self.approximate_posterior is None:
            # Assume independent latents and that they have the same dimension
            self.approximate_posterior = MeanFieldApproximatePosterior(
                dim_list=[self.prior.latents[0].sparsity.Z.shape[0]]*self.prior.num_latents
            )

    def get_objective(self):

        elbo = self.inference.ELBO(
            self.data,
            self.likelihood,
            self.prior,
            self.approximate_posterior
        )

        return -elbo

    def mean(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False)

        mu = np.reshape(mu, [self.output_dim, XS.shape[0], 1])
        return mu

    def var(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)

        #chex.assert_shape(var, [self.output_dim, XS.shape[0], 1])
        var = np.reshape(var, [self.output_dim, XS.shape[0], 1])
        return var

    def covar(self, XS_1, XS_2, X=None, Y=None):
        var_arr =  self.inference.predictive_covar(
            XS_1, 
            XS_2, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior
        )

        chex.assert_shape(var_arr, [self.output_dim, XS_1.shape[0], XS_2.shape[0]])
        return var_arr

    def predict_latents(self, XS, diagonal=True, squeeze=True):
        mean, var = self.inference.predict_latents(
            XS, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )

        # TODO: make consistent with predict_f/y
        # TODO: make work is diagonal=False
        # ensure shap is [N, P]
        P = mean.shape[-1]
        N = XS.shape[0]

        if False:
            mean = np.reshape(mean, [N, P])
            var = np.reshape(var, [N, P])

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var

    def predict_f(self, XS, diagonal=True, squeeze=True):

        mean, var = self.inference.predict_f(
            XS, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )

        # ensure shap is [P, N]
        P = self.prior.output_dim
        N = XS.shape[0]


        mean = np.reshape(mean, [P, N])
        var = np.reshape(var, [P, N])

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var

    def predict_y(self, XS, diagonal=True, squeeze=True):
        mu_arr, var_arr =  self.inference.predict_y(
            XS, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )
        # ensure shap is [P, N]
        P = mu_arr.shape[0]
        N = XS.shape[0]

        mu_arr = np.reshape(mu_arr, [P, N])
        var_arr = np.reshape(var_arr, [P, N])

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr) 

        return mu_arr, var_arr

    def natural_gradient_update(self, learning_rate, enforce_psd_type=None):
        natgrad_fn = evoke('natural_gradients', self, self.approximate_posterior)

        return natgrad_fn(
            self,
            learning_rate,
            enforce_psd_type
        )

    def natural_gradient(self, learning_rate):
        raise NotImplementedError()

    def confidence_intervals(self, XS):
        """ Returns the median and the 95% confidence intervals. """
        return evoke('confidence_intervals', self)(XS, self)
