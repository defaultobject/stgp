from .. import Likelihood
from .. import Distribution
from ..dispatcher import Dispatcher

from ..likelihoods import GaussianLikelihood
from ..approximate_posteriors import *
from ..distributions import *

from .general import log_chol_matrix_det, cholesky_solve
from ..settings import Settings

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial

import typing
from typing import List

@Dispatcher.register('kullback_leiblers', KernelGaussianDistribution)
@Dispatcher.register('kullback_leiblers', BlockGaussianDistribution)
#@jit
def gaussian_kl(X:np.ndarray, g1: GaussianDistribution, g2: GaussianDistribution) -> np.ndarray:
    """
        computs KL[g1, g2]
    """

    mu_1 = g1.mean(X)
    mu_2 = g2.mean(X)

    covar_chol_1 = g1.covar_chol(X, X)
    covar_chol_2 = g2.covar_chol(X, X)

    covar_1 = g1.covar(X, X)
    covar_2 = g2.covar(X, X)


    log_det_term = log_chol_matrix_det(covar_chol_2) - log_chol_matrix_det(covar_chol_1)

    #see https://github.com/GPflow/GPflow/blob/develop/gpflow/kullback_leiblers.py 
    trace_term = np.sum(np.square(jax.scipy.linalg.solve_triangular(covar_chol_2, covar_chol_1, lower=True)))
    #trace_term = np.trace(cholesky_solve(covar_chol_2, covar_1))

    err = mu_2-mu_1 
    #maha_term = np.matmul(err.T, cholesky_solve(covar_chol_2, err))
    maha_term = np.sum(np.square(jax.scipy.linalg.solve_triangular(covar_chol_2, err, lower=True)))

    N = mu_1.shape[0]*1.0


    return 0.5 * (log_det_term - N + trace_term + maha_term)

@Dispatcher.register('kullback_leiblers', BlockWhitenedGaussianDistribution)
@Dispatcher.register('kullback_leiblers', WhitenedKernelGaussianDistribution)
#@jit
def whitened_gaussian_kl(X:np.ndarray, g1: GaussianDistribution, g2: GaussianDistribution) -> np.ndarray:
    """
        Assumes that g2 is a standard Gaussian - N(0, I) 
    """
    mu_1 = g1.mean(X)
    covar_chol_1 = g1.covar_chol(X, X)
    covar_1 = g1.covar(X, X)

    log_det_term = - log_chol_matrix_det(covar_chol_1)

    trace_term = np.trace(covar_1)

    maha_term = np.sum(np.square(mu_1))

    N = mu_1.shape[0]*1.0

    return 0.5 * (log_det_term - N + trace_term + maha_term )


