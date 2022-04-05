import jax
from jax import jit
import jax.numpy as np
import objax
import chex
from typing import List
from objax import ModuleList

from ...import settings
from ...core import GPPrior
from ...likelihood import Gaussian
from ...approximate_posteriors import GaussianApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior
from ...dispatch import dispatch
from ..gaussian import log_gaussian
from ..matrix_ops import add_jitter, cholesky, cholesky_solve
from ... import utils


@jit
def scalar_gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar_diag:np.ndarray) ->  np.ndarray:

    chex.assert_equal(X.shape[0], 1)
    chex.assert_shape(Y, [1, 1])
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    N = Y.shape[0]
    c1 = -0.5*np.log(2*np.pi) - 0.5*np.log(noise)

    err = Y - q_mu
    err = np.sum(np.matmul(err.T, err))

    ell =  N*c1  -0.5*(err + np.sum(q_covar_diag))/noise

    chex.assert_rank(ell, 0)
    return ell

@jit
def gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar_diag:np.ndarray) ->  np.ndarray:

    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    N = Y.shape[0]
    c1 = -0.5*np.log(2*np.pi) - 0.5*np.log(noise)

    err = Y - q_mu
    err = np.sum(np.matmul(err.T, err))

    ell =  N*c1  -0.5*(err + np.sum(q_covar_diag))/noise


    chex.assert_rank(ell, 0)
    return ell

@jit
def diagonal_gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar_diag:np.ndarray) ->  np.ndarray:
    """
        Args:
            X: N x D
            Y: N x 1
            noise: N x 1
            q_mu: N x 1
            q_covar_diag: N x 1
    """

    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    N = Y.shape[0]
    c1 = -0.5*N*np.log(2*np.pi) - 0.5*np.sum(np.log(noise))

    err = Y - q_mu
    inv_noise = 1/noise
    err = np.sum(np.matmul(err.T, np.multiply(inv_noise, err)))

    ell =  c1  -0.5*(err + np.sum(np.multiply(inv_noise, q_covar_diag)))

    chex.assert_rank(ell, 0)
    return ell

@jit
def full_gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar:np.ndarray) ->  np.ndarray:
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(q_covar,  2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_shape(q_covar, [Y.shape[0], Y.shape[0]])

    noise_chol = cholesky(add_jitter(noise, settings.jitter))

    ml =  log_gaussian(Y, q_mu, noise) 
    trace_term = -0.5*np.trace(cholesky_solve(noise_chol, q_covar))

    ell =  ml + trace_term

    chex.assert_rank(ell, 0)
    return ell

@jit
def scalar_poisson_expected_log_likelihood(X:np.ndarray, Y:np.ndarray,  binsize:float, q_mu:np.ndarray, q_covar_diag:np.ndarray) -> np.ndarray:
    """
        X, Y, q_mu, q_covar are all scalars
        Let a = E[f] = m and b = E[exp(f)] = exp(m+v/2) then
            E[log Poisson(y | exp(f)*binsize)] = Y log binsize  + E[Y * log exp(f)] - E[binsize * exp(f)] - log Y!
                                               = Y log binsize + Y * m - binsize * exp(m + v/2) - log Y!
    """

    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    Y = np.squeeze(Y)
    q_mu = np.squeeze(q_mu)
    binsize = np.squeeze(binsize)

    ell =  Y*np.log(binsize) + Y*q_mu - binsize*np.exp(q_mu + q_covar/2) - gammaln(Y+1.0)

    chex.assert_rank(ell, 0)
    return ell


