"""
    Implementations of closed form marginal likelihoods
"""
from ..likelihoods import *
from ..approximate_posteriors import *
from ..distributions import *
from ..data import Data, ListData

from .general import log_chol_matrix_det, cholesky_solve
from ..settings import Settings

from ..computation.utils import get_reparameterised_lmc_prior, get_blocks_from_lmc_proir
from ..computation.gaussian import log_gaussian

from ..dispatcher import Dispatcher

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial

import numpy as onp

import typing
from typing import List


#@jit
@Dispatcher.register('marginal_likelihood', GaussianLikelihood, KernelGaussianDistribution)
def gaussian_gaussian_marginal_likelihood(X: np.ndarray, Y: np.ndarray, likelihood: GaussianLikelihood, prior: KernelGaussianDistribution) -> np.ndarray:
    #Marginal likleihood of a gaussian prior and gaussian likelihood 
    N = X.shape[0]

    kernel = prior.kernel
    k_xx = kernel.K(X, X)
    lik_noise = likelihood.variance
    k = k_xx + lik_noise * np.eye(N)

    print('lik_noise: ', lik_noise)

    return log_gaussian(Y, np.zeros_like(Y), k)

#@jit
@Dispatcher.register('marginal_likelihood', NaturalDiagonalGaussianLikelihood, KernelGaussianDistribution)
def natural_diagonal_gaussian_gaussian_marginal_likelihood(X: np.ndarray, Y: np.ndarray, likelihood: DiagonalGaussianLikelihood, prior: KernelGaussianDistribution) -> np.ndarray:
    #Marginal likleihood of a gaussian prior and gaussian likelihood 
    N = X.shape[0]

    Y = likelihood.Y
    kernel = prior.kernel
    k_xx = kernel.K(X, X) # N x N
    lik_noise = np.squeeze(likelihood.variance) # N 
    k = k_xx + np.diag(lik_noise) # N x N

    return log_gaussian(Y, np.zeros_like(Y), k)

@Dispatcher.register('marginal_likelihood', NaturalBlockDiagonalGaussianLikelihood, KernelGaussianDistribution)
def natural_block_diagonal_gaussian_gaussian_marginal_likelihood(X: np.ndarray, Y: np.ndarray, likelihood: DiagonalGaussianLikelihood, prior: KernelGaussianDistribution) -> np.ndarray:
    #Marginal likleihood of a gaussian prior and gaussian likelihood 
    N = X.shape[0]

    Y = likelihood.Y
    Y = np.reshape(Y, [Y.shape[0]*Y.shape[1], 1])
    lik_var = jax.scipy.linalg.block_diag(*likelihood.variance)
    kernel = prior.kernel
    k_xx = kernel.K(X, X) # N x N
    k = k_xx + lik_var # N x N

    return log_gaussian(Y, np.zeros_like(Y), k)

@Dispatcher.register('marginal_likelihood', DiagonalGaussianLikelihood, KernelGaussianDistribution)
def diagonal_gaussian_gaussian_marginal_likelihood(X: np.ndarray, Y: np.ndarray, likelihood: DiagonalGaussianLikelihood, prior: KernelGaussianDistribution) -> np.ndarray:
    #Marginal likleihood of a gaussian prior and gaussian likelihood 
    N = X.shape[0]

    kernel = prior.kernel
    k_xx = kernel.K(X, X) # N x N
    lik_noise = np.squeeze(likelihood.variance) # N 
    k = k_xx + np.diag(lik_noise) # N x N

    return log_gaussian(Y, np.zeros_like(Y), k)

@Dispatcher.register('marginal_likelihood', LMC_Constrained_Likelihood, KernelGaussianDistribution)
@Dispatcher.register('marginal_likelihood', LMC_Likelihood, KernelGaussianDistribution)
def gaussian_lmc_marginal_likelihood(data:ListData, likelihood: LMC_Likelihood, prior_arr: List[KernelGaussianDistribution]) -> np.ndarray:
    """
        The LMC marginal likelihood is just a Gaussian marginal likelihood with the prior stacked into a block diagional form.
        All X_arr and Y_arr MUST have the same dimension
        
        Model:
            The LMC Model implemented is defined as:

                X_arr = [X_p] where X_p ∈ N x D
                Y_arr = [Y_p] where Y_p ∈ N x 1
                
                Y = vec(Y_arr)
                X = vec(X_arr)

            with prior:

                p(F) = N(F| 0, (W ⊗ I) blkdiag(K_1(X, X), ..., K_P(X, X)) (W ⊗ I)^T)

            The LMC Likelihood is:

                p(Y|F) = N(Y | F, diag(sigma_1, ..., sigma_P))
            
            Resulting in the following marginal likelihood:

                p(Y) = N( Y | 0, (W ⊗ I) blkdiag(K_1(X, X), ..., K_P(X, X)) (W ⊗ I)^T +  diag(sigma_1, ..., sigma_P))

        Dealing with missing data:
            Y is a vector with missing observations represented by NaNs. 
            
    """
    X_arr = data.X
    Y_arr = data.Y

    num_latents = len(prior_arr)
    num_outputs = len(X_arr)

    #collect mixing weights
    variances = likelihood.variances()
    mixing_weights = likelihood.coregion_weights

    #assume X_arr all have same dimension
    K = get_blocks_from_lmc_proir(X_arr, X_arr, likelihood, prior_arr)
    W = np.kron(mixing_weights, np.eye(Y_arr[0].shape[0]))

    variance_arr = [variances[output]*np.eye(X_arr[output].shape[0]) for output in range(num_outputs)]
    variance_arr = jax.scipy.linalg.block_diag(*variance_arr)

    sigma = W @ K @ W.T + variance_arr

    Y = np.vstack(Y_arr)

    bool_mask = np.logical_not(np.hstack(data.mask))
    Y = Y[bool_mask, :]
    sigma = sigma[bool_mask, ...]
    sigma = sigma[..., bool_mask]

    ml =  log_gaussian(Y, np.zeros_like(Y), sigma)

    return ml


