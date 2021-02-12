from . import ApproximatePosterior
from . import GaussianApproxPosterior

from .. import Likelihood
from .. import Distribution
from .. import Dispatcher

from ..data import Data, ListData

from ..likelihoods import LMC_Likelihood
from ..distributions import GaussianDistribution
from ..settings import Settings

from ..computation.conditionals import *
from ..computation.kullback_leiblers import *
from ..computation.expectation_approximators import *

from ..decorators import *

import jax
import jax.numpy as np
from jax import jit, partial

import typing
from typing import Optional, List, Union, Tuple


import warnings

class MeanFieldApproxPosterior(ApproximatePosterior):
    def __init__(self, components: Optional[List[Distribution]]=None, key:Optional[jax.random.PRNGKey]=None, name:Optional[str]=None):
        """
            Args:
                m: np.ndarray inital mean values for q(.)
                S: np.ndarray inital variance values for q(.)
        """

        #save __init__ arguments as properties of this object
        self.save_inputs_to_properties(locals())

        #maps for special cases to specific functions used to compute each case

        if False:
            self.ell_special_cases = {
                LMC_Likelihood: meanfield_lmc_expected_log_likeliood ,
                LMC_Constrained_Likelihood: meanfield_lmc_expected_log_likeliood 
            }

            self.predict_special_cases = {
                LMC_Likelihood: [None, meanfield_lmc_predict_y_diagional],
                LMC_Constrained_Likelihood: [None, meanfield_lmc_predict_y_diagional]
            }

        self.num_samples = 5
        self.num_prediction_samples = 20
 
        name = name or self.__class__.__name__ 

        super(MeanFieldApproxPosterior, self).__init__(name=self.name)

    def setup(self, N_latents: List[int]) -> None:
        """
            Args:
                N_latents: a list of dimensions for each component of the mean field approximate posterior
        """
        if self.components is None:
            if Settings.strict_mode:
                raise RuntimeError('MeanFieldApproxPosterior is not initalised')

            warnings.warn('MeanFieldApproxPosterior is not initalised. Default will be used.')

            components = []
            #no variational_posterior has been setup yet
            self.num_latents = len(N_latents)
            self.N_latents = N_latents
            for n in self.N_latents:
                q_p = GaussianApproxPosterior(dim=n)
                q_p.setup()

                components.append(q_p)

            self.components = components

        if self.key is None:
            self.key = jax.random.PRNGKey(Settings.seed)

    def expected_log_likelihood(self, data: Data, likelihood: Likelihood, model: 'Model') -> np.ndarray:

        fun = Dispatcher.dispatch('expected_log_likelihoods', type(likelihood), type(self))

        if fun is not False:
            ell = fun(
                data,
                likelihood,
                model,
                self
            )
       
            return ell
        else:
            ell =  mean_field_gauss_hermite_quadrature(
                data,
                likelihood,
                model,
                self.num_samples
            )
            return ell


    def predict_y(self, data_xs: Data, data: Data, model: 'Model', diagonal_var) -> Tuple[np.ndarray, np.ndarray]:
        XS = data_xs.X

        likelihood = model.likelihood
        mean_arr, var_arr =  self.predict_f(XS, data, model, diagonal_var, predict=True)


        if likelihood.predict_requires_approximation():
            return predict_mean_field_gauss_hermite_quadrature(
                data_xs,
                data,
                likelihood,
                model,
                self.num_prediction_samples
            )
        else:
            return likelihood.predict_mean_var(mean_arr, var_arr)

    @ensure_data_passed
    def predict_f(self, data_xs: Data, data: Data, model: 'Model', diagonal_var, predict=False) -> Tuple[np.ndarray, np.ndarray]:

        num_latents = len(self.components)

        XS = data_xs.X

        latent_mu_arr = []
        latent_sig_arr = []
        for latent in range(num_latents):
            mu, diag_var = self.components[latent].predict_f(XS, data, model, latent, diagonal_var=diagonal_var, predict=predict)
            latent_mu_arr.append(mu)
            latent_sig_arr.append(diag_var)

        return latent_mu_arr, latent_sig_arr

    def KL(self, X: List[np.ndarray], distribution_arr: List[Distribution]) -> np.ndarray:
        """
            For a mean field approximation the KL is simply the sum of KL terms
                for each component
        """
        kl_sum = 0.0
        Z = X[0]
        for latent, component in enumerate(self.components):
            kl_sum += component.KL(Z, distribution_arr[latent])
        return kl_sum
