from . import ApproximatePosterior
from . import GaussianApproxPosterior

from .. import Likelihood
from .. import Distribution

from .. import Data

from ..likelihoods import LMC_Likelihood
from ..distributions import GaussianDistribution, BlockGaussianDistribution, WhitenedKernelGaussianDistribution
from ..settings import Settings

from ..computation.conditionals import *
from ..computation.kullback_leiblers import *
from ..computation.expected_log_likelihoods import *
#from ..computation.predictors import *

import jax
import jax.numpy as np
from jax import jit, partial

import typing
from typing import Optional, List, Union, Tuple


import warnings

class FullStructuredApproxPosterior(ApproximatePosterior):
    def __init__(self, distribution: Optional[Distribution]=None, key:Optional[jax.random.PRNGKey]=None, name:Optional[str]=None):
        """
            Args:
                m: np.ndarray inital mean values for q(.)
                S: np.ndarray inital variance values for q(.)
        """

        #save __init__ arguments as properties of this object
        self.save_inputs_to_properties(locals())

        #maps for special cases to specific functions used to compute each case


        self.ell_special_cases = {
            LMC_Likelihood: full_structured_lmc_expected_log_likeliood ,
            LMC_Constrained_Likelihood: full_structured_lmc_expected_log_likeliood 
        }

        self.predict_special_cases = {
            LMC_Likelihood: [None, full_structured_lmc_predict_y_diagional],
            LMC_Constrained_Likelihood: [None, full_structured_lmc_predict_y_diagional]
        }

        self.num_samples = 100
        self.num_prediction_samples = 1000
 

        name = name or self.__class__.__name__ 

        super(FullStructuredApproxPosterior, self).__init__(name=self.name)

    def setup(self, N: int) -> None:
        """
            Args:
                N: dimension of approximate posterior
        """

        if self.distribution is None:
            if Settings.strict_mode:
                raise RuntimeError('FullStructuredApproxPosterior is not initalised')

            warnings.warn('FullStructuredApproxPosterior is not initalised. Default will be used.')

            self.distribution = GaussianApproxPosterior(dim=N)

        if self.key is None:
            self.key = jax.random.PRNGKey(Settings.seed)


    @jit
    def expected_log_likelihood(self, data:Data, prior: Distribution, likelihood: Likelihood, sparsity:Sparsity) -> np.ndarray:
        X, Y = data.X, data.Y

        for key, func in  self.ell_special_cases.items():
            if isinstance(likelihood, key):
                ell = func(
                    data, 
                    prior,
                    likelihood,
                    sparsity,
                    self
                )
               
                return ell


        raise NotImplementedError();

    def predict_y(self, XS:np.ndarray, prior:Distribution, likelihood:Likelihood, sparsity_arr: List[Sparsity], diagional_var) -> Tuple[np.ndarray, np.ndarray]:

        for key, func in  self.predict_special_cases.items():
            func_full, func_diag = func[0], func[1]
            if type(likelihood) == key:
                if diagional_var:
                    return func_diag(
                            XS,
                            prior,
                            likelihood,
                            sparsity_arr,
                            self
                    )
        raise NotImplementedError();

    def predict_latents(self, XS:np.ndarray, prior:Distribution, likelihood:Likelihood, sparsity_arr, diagional_var) -> Tuple[List, List]:
        prior_arr = prior
        if type(prior_arr[0]) is WhitenedKernelGaussianDistribution:
            #the prior is whitened
            block_prior = BlockWhitenedGaussianDistribution(prior_arr)
        else:
            block_prior = BlockGaussianDistribution(prior_arr)

        Q = len(prior)
        N = XS.shape[0]

        mu_arr, var_arr = [], []

        #TODO: assumed sparsity_arr
        mu, var_diag = self.distribution.predict_f(XS, block_prior, sparsity_arr[0], diagional_var=True)
        mu = np.reshape(mu, [mu.shape[0], 1])
        var_diag = np.reshape(var_diag, [var_diag.shape[0], 1])

        for q in range(Q):
            mu_arr.append(mu[N*q:N*(q+1), 1])
            var_arr.append(var_diag[N*q:N*(q+1), 1])

        return mu_arr, var_arr

    def KL(self, X: List[np.ndarray], distribution_arr: List[Distribution]) -> np.ndarray:
        """
            For a mean field approximation the KL is simply the sum of KL terms
                for each component
            #TODO create a blockdiagional Gaussian distribution from a list of gaussians and add to KL
            #TOOD: look at how pytorch does mixins - see https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py
            #TODO: move block diagional matrix into the model class, so that we can remove distribution_arr
        """
        if type(distribution_arr[0]) is WhitenedKernelGaussianDistribution:
            #the prior is whitened
            block_prior = BlockWhitenedGaussianDistribution(distribution_arr)
        else:
            block_prior = BlockGaussianDistribution(distribution_arr)

        X_stacked = np.vstack(X)
        KL =  self.distribution.KL(X_stacked, block_prior)
        return KL

