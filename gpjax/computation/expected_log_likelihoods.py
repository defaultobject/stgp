from ..sparsity import Sparsity
from ..dispatcher import Dispatcher
from ..data import Data

from ..likelihoods import *
from ..approximate_posteriors import ApproximatePosterior

from ..approximate_posteriors.gaussian_approx_posterior import GaussianApproxPosterior
from ..approximate_posteriors.mean_field_approx_posterior import MeanFieldApproxPosterior
from ..approximate_posteriors.diagonal_conjugate_approx_posterior import DiagonalConjugateApproxPosterior
from ..approximate_posteriors.block_diagonal_conjugate_approx_posterior import BlockDiagonalConjugateApproxPosterior

from .general import cholesky_solve, log_chol_matrix_det, cholesky

from .gaussian import log_gaussian


from ..settings import Settings
from ..data import Data, ListData

from ..distributions import *

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial
from jax.scipy.special import erf, gammaln

import typing
from typing import List

@jit
def gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar_diag:np.ndarray) ->  np.ndarray:

    N = Y.shape[0]
    c1 = -0.5*np.log(2*np.pi) - 0.5*np.log(noise)

    err = Y - q_mu
    err = np.sum(np.matmul(err.T, err))

    return N*c1  -0.5*(err + np.sum(q_covar_diag))/noise

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
    N = Y.shape[0]
    c1 = -0.5*N*np.log(2*np.pi) - 0.5*np.sum(np.log(noise))

    err = Y - q_mu
    inv_noise = 1/noise
    err = np.sum(np.matmul(err.T, np.multiply(inv_noise, err)))

    return c1  -0.5*(err + np.sum(np.multiply(inv_noise, q_covar_diag)))

@jit
def full_gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar:np.ndarray) ->  np.ndarray:
    """
        Args:
            X: N x D
            Y: N x 1
            noise: N x N
            q_mu: N x 1
            q_covar_diag: N x 1
    """

    sigma_chol = cholesky(noise+Settings.jitter*np.eye(noise.shape[0]))

    ml =  log_gaussian(Y, q_mu, noise) 
    trace_term = -0.5*np.trace(cholesky_solve(sigma_chol, q_covar))

    return ml + trace_term

@jit
def scalar_poisson_expected_log_likelihood(X:np.ndarray, Y:np.ndarray,  binsize:float, q_mu:np.ndarray, q_covar:np.ndarray) -> np.ndarray:
    """
        X, Y, q_mu, q_covar are all scalars

        Let a = E[f] = m and b = E[exp(f)] = exp(m+v/2) then

            E[log Poisson(y | exp(f)*binsize)] = Y log binsize  + E[Y * log exp(f)] - E[binsize * exp(f)] - log Y!
                                               = Y log binsize + Y * m - binsize * exp(m + v/2) - log Y!
    """
    Y = np.squeeze(Y)
    q_mu = np.squeeze(q_mu)
    binsize = np.squeeze(binsize)

    return Y*np.log(binsize) + Y*q_mu - binsize*np.exp(q_mu + q_covar/2) - gammaln(Y+1.0)

@Dispatcher.register('precomputed_expected_log_likelihoods', PoissonLikelihood, DiagonalConjugateApproxPosterior)
@Dispatcher.register('precomputed_expected_log_likelihoods', PoissonLikelihood, GaussianApproxPosterior)
def precomputed_gaussian_poisson_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, q_mu, q_covar_diag, latent: int, q:ApproximatePosterior) ->  np.ndarray:
    X = data.X[latent]
    Y = data.Y[latent]
    binsize = likelihood.binsize

    mask = data.mask

    if type(mask) is list:
        mask = mask[0]

    if True:
        bool_mask = np.logical_not(mask)
        X = X[bool_mask, :]
        Y = Y[bool_mask, :]
        q_mu = q_mu[bool_mask, :]
        q_covar_diag = q_covar_diag[bool_mask, :]

    ell_vmap = jax.vmap(scalar_poisson_expected_log_likelihood, (0, 0, None, 0, 0))

    return np.sum(ell_vmap(X, Y, binsize, q_mu, q_covar_diag))



@Dispatcher.register('precomputed_expected_log_likelihoods', PoissonLikelihood, BlockDiagonalConjugateApproxPosterior)
def precomputed_st_gaussian_poisson_expected_log_likelihood(data: Data, likelihood:PoissonLikelihood, q_mu, q_covar_diag, latent: int, q:ApproximatePosterior) ->  np.ndarray:
    X = data.flattened_X[latent]
    Y = data.flattened_Y[latent]

    binsize = likelihood.binsize

    mask = data.mask
    if type(mask) is list:
        mask = mask[0]

    if True:
        mask = data.flattened_mask[0]
        bool_mask = np.logical_not(mask)

        X = X[bool_mask, :]
        Y = Y[bool_mask, :]
        q_mu = q_mu[bool_mask, :]
        q_covar_diag = q_covar_diag[bool_mask, :]


    ell_vmap = jax.vmap(scalar_poisson_expected_log_likelihood, (0, 0, None, 0, 0))

    return np.sum(ell_vmap(X, Y, binsize, q_mu, q_covar_diag))

@Dispatcher.register('expected_log_likelihoods', PoissonLikelihood, DiagonalConjugateApproxPosterior)
@Dispatcher.register('expected_log_likelihoods', PoissonLikelihood, GaussianApproxPosterior)
def gaussian_poisson_expected_log_likelihood(data: Data, likelihood:PoissonLikelihood, model: 'Model', latent: int, q:ApproximatePosterior) ->  np.ndarray:
    _X = data.X[latent]
    q_mu, q_covar_diag = q.predict_f(_X, data, model, latent, diagonal_var=True)


    return precomputed_gaussian_poisson_expected_log_likelihood(data, likelihood, q_mu, q_covar_diag, latent, q)

@Dispatcher.register('expected_log_likelihoods', GaussianLikelihood, DiagonalConjugateApproxPosterior)
@Dispatcher.register('expected_log_likelihoods', GaussianLikelihood, GaussianApproxPosterior)
def gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, model: 'Model', latent: int, q:ApproximatePosterior) ->  np.ndarray:

    X = data.X[latent]
    Y = data.Y[latent]
    noise = likelihood.variance
    q_mu, q_covar_diag = q.predict_f(X, data, model, latent, diagonal_var=True)


    if True:
        mask = data.mask
        if type(mask) is list:
            mask = mask[0]
        bool_mask = np.logical_not(mask)
        X = X[bool_mask, :]
        Y = Y[bool_mask, :]
        q_mu = q_mu[bool_mask, :]
        q_covar_diag = q_covar_diag[bool_mask, :]

    return gaussian_expected_log_likelihood(X, Y, noise, q_mu, q_covar_diag)

@Dispatcher.register('precomputed_expected_log_likelihoods', GaussianLikelihood, DiagonalConjugateApproxPosterior)
@Dispatcher.register('precomputed_expected_log_likelihoods', GaussianLikelihood, GaussianApproxPosterior)
def precomputed_gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, q_mu, q_covar_diag, latent: int, q:ApproximatePosterior) ->  np.ndarray:
    X = data.X[latent]
    Y = data.Y[latent]
    noise = likelihood.variance

    return gaussian_expected_log_likelihood(X, Y, noise, q_mu, q_covar_diag)

@Dispatcher.register('expected_log_likelihoods', GaussianLikelihood, BlockDiagonalConjugateApproxPosterior)
def st_gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, model: 'Model', latent: int, q:ApproximatePosterior) ->  np.ndarray:
    X = data.flattened_X[latent]
    Y = data.flattened_Y[latent]

    noise = likelihood.variance
    q_mu, q_covar_diag = q.predict_f(data, data, model, latent, diagonal_var=True, spatial_predict=True, temporal_predict=False)

    if True:
        bool_mask = np.logical_not(data.flattened_mask[0])
        X = X[bool_mask, :]
        Y = Y[bool_mask, :]
        q_mu = q_mu[bool_mask, :]
        q_covar_diag = q_covar_diag[bool_mask, :]

    return gaussian_expected_log_likelihood(X, Y, noise, q_mu, q_covar_diag)


@Dispatcher.register('precomputed_expected_log_likelihoods', GaussianLikelihood, BlockDiagonalConjugateApproxPosterior)
def precomputed_st_gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, q_mu, q_covar_diag, latent: int, q:ApproximatePosterior) ->  np.ndarray:
    X = data.flattened_X[latent]
    Y = data.flattened_Y[latent]

    noise = likelihood.variance

    if True:
        bool_mask = np.logical_not(data.flattened_mask[0])
        X = X[bool_mask, :]
        Y = Y[bool_mask, :]
        q_mu = q_mu[bool_mask, :]
        q_covar_diag = q_covar_diag[bool_mask, :]

    ell = gaussian_expected_log_likelihood(X, Y, noise, q_mu, q_covar_diag)
    return ell

@Dispatcher.register('expected_log_likelihoods', DiagonalGaussianLikelihood, DiagonalConjugateApproxPosterior)
@Dispatcher.register('expected_log_likelihoods', DiagonalGaussianLikelihood, GaussianApproxPosterior)
def diagonal_gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, model: 'Model', latent: int, q:ApproximatePosterior) ->  np.ndarray:

    X = data.X[latent]
    Y = data.Y[latent]

    noise = likelihood.variance
    q_mu, q_covar_diag = q.predict_f(X, data, model, latent, diagonal_var=True)

    return diagonal_gaussian_expected_log_likelihood(X, Y, noise, q_mu, q_covar_diag)

@Dispatcher.register('expected_log_likelihoods', NaturalBlockDiagonalGaussianLikelihood, BlockDiagonalConjugateApproxPosterior)
def block_diagonal_gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, model: 'Model', latent: int, q:ApproximatePosterior) ->  np.ndarray:

    X = data.X[latent]
    Y = data.Y[latent]


    Y = likelihood.Y
    noise = likelihood.variance

    q_mu, q_covar= q.predict_f(X, data, model, latent=latent, diagonal_var=False, spatial_predict=False, temporal_predict=False)

    N, D = Y.shape[0], Y.shape[1]

    q_mu = q_mu.reshape([N, D, 1])
    q_covar = q_covar.reshape([N, D, D])

    batched_ell = jax.vmap(full_gaussian_expected_log_likelihood, (None, 0, 0, 0, 0), (0))

    ell = batched_ell(X, Y, noise, q_mu, q_covar)

    return np.sum(ell)

@Dispatcher.register('precomputed_expected_log_likelihoods', NaturalBlockDiagonalGaussianLikelihood, BlockDiagonalConjugateApproxPosterior)
def precomputed_block_diagonal_gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, q_mu, q_covar, latent: int, q:ApproximatePosterior) ->  np.ndarray:

    X = data.X[latent]
    Y = data.Y[latent]


    Y = likelihood.Y
    noise = likelihood.variance

    N, D = Y.shape[0], Y.shape[1]

    q_mu = q_mu.reshape([N, D, 1])
    q_covar = q_covar.reshape([N, D, D])

    batched_ell = jax.vmap(full_gaussian_expected_log_likelihood, (None, 0, 0, 0, 0), (0))

    ell = batched_ell(X, Y, noise, q_mu, q_covar)

    return np.sum(ell)


@Dispatcher.register('expected_log_likelihoods', NaturalDiagonalGaussianLikelihood, DiagonalConjugateApproxPosterior)
@Dispatcher.register('expected_log_likelihoods', DiagonalGaussianLikelihood, GaussianApproxPosterior)
def natural_diagonal_gaussian_gaussian_expected_log_likelihood(data: Data, likelihood:GaussianLikelihood, model: 'Model', latent: int, q:ApproximatePosterior) ->  np.ndarray:

    X = data.X[latent]
    Y = data.Y[latent]

    Y = likelihood.Y
    noise = likelihood.variance
    q_mu, q_covar_diag = q.predict_f(X, data, model, latent, diagonal_var=True)


    return diagonal_gaussian_expected_log_likelihood(X, Y, noise, q_mu, q_covar_diag)

@Dispatcher.register('expected_log_likelihoods', LMC_Constrained_Likelihood, MeanFieldApproxPosterior)
@Dispatcher.register('expected_log_likelihoods', LMC_Likelihood, MeanFieldApproxPosterior)
def meanfield_lmc_expected_log_likeliood(data:ListData, likelihood:LMC_Likelihood, model: 'Model', q:ApproximatePosterior) -> np.ndarray:
    X = data.X
    Y = data.Y

    num_latents = model.num_latents
    num_outputs = len(X)

    latent_mu_arr = []
    latent_sig_arr = []

    mixing_weights = likelihood.coregion_weights
    variance = likelihood.variance

    for latent in range(num_latents):
        mu, diag_var = q.components[latent].predict_f(X[latent], data, model, latent, diagonal_var=True)
        latent_mu_arr.append(mu)
        latent_sig_arr.append(diag_var)

    non_nan_masks = data.mask

    ell = 0.0
    for output in range(num_outputs):
        output_weights = mixing_weights[output, :] #P
        mu = np.sum([output_weights[q]*latent_mu_arr[q] for q in range(num_latents)], axis=0) #Nx1
        sig = np.sum([(output_weights[q]**2)*latent_sig_arr[q] for q in range(num_latents)], axis=0) #Nx1

        #remove nan observations
        Y_p = Y[output]

        bool_mask = np.logical_not(data.mask[output])
        Y_p = Y_p[bool_mask, :]
        mu = mu[bool_mask, :]
        sig = sig[bool_mask, :]

        ell += gaussian_expected_log_likelihood(X[output], Y_p, variance[output], mu, sig)

    return ell

def extract_blocks(X:np.ndarray, blocksize:int, keep_as_view=False):
    #from https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array
    M,N = X.shape
    b0, b1 = blocksize, blocksize

    if keep_as_view==0:
        return np.reshape(np.swapaxes(np.reshape(X, [M//b0,b0,N//b1,b1]), 1, 2), [-1,b0,b1])
    else:
        return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2)

def full_structured_lmc_expected_log_likeliood(data:ListData, prior_arr: List[Distribution], likelihood:LMC_Likelihood, sparsity_arr: List[Sparsity], q:ApproximatePosterior) -> np.ndarray:
    X, Y = data.X, data.Y

    #TODO: use predict_f

    if type(prior_arr[0]) is WhitenedKernelGaussianDistribution:
        #the prior is whitened
        block_prior = BlockWhitenedGaussianDistribution(prior_arr)
    else:
        block_prior = BlockGaussianDistribution(prior_arr)

    #TODO: assume same inducing points for now
    m, S = q.distribution.predict_f(X[0], block_prior, sparsity_arr[0], dagional_var=False)

    P = likelihood.num_outputs
    Q = likelihood.num_latents
    N = X[0].shape[0]

    Y_vec = np.vstack(Y)

    mixing_weights = likelihood.coregion_weights
    variance = likelihood.variance


    latent_mu_arr = np.reshape(m, [Q, N, 1])

    non_nan_masks = data.mask

    ell = 0.0
    for output in range(P):
        output_weights = mixing_weights[output, :]
        mu = np.sum([output_weights[q]*latent_mu_arr[q] for q in range(Q)], axis=0)

        output_weights = np.reshape(mixing_weights[output, :], [Q, 1])
        W = np.kron(output_weights @ output_weights.T, np.eye(N))
        S_output = np.multiply(S, W)
        #extract vector of blocks of size N, N
        S_output = extract_blocks(S_output, blocksize=N) #Q^2 x N x N
        #extract the diagional vectors from each of the Q^2 blocks
        S_blkdiag = np.diagonal(S_output, axis1=1, axis2=2) #Q^2 x N

        sig =  np.reshape(np.sum(S_blkdiag, axis=0), [N, 1])

        #remove nan observations
        Y_p = np.take(Y[output], non_nan_masks[output], axis=0)
        mu = np.take(mu, non_nan_masks[output], axis=0)
        sig = np.take(sig, non_nan_masks[output], axis=0)

        ell += gaussian_expected_log_likelihood(X[output], Y_p, variance[output], mu, sig)

    return ell
