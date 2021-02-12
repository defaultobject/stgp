from .. import Likelihood
from .. import Distribution
from .. import Sparsity
from ..data import Data, ListData

from ..models import *
from ..likelihoods import *
from ..approximate_posteriors import *
from ..distributions import *
from ..computation.utils import get_reparameterised_lmc_prior, get_reparameterised_lmc_prior_diagional, get_blocks_from_lmc_proir, get_blocks_from_lmc_proir_diagional

from .general import log_chol_matrix_det, cholesky_solve, cholesky
from ..settings import Settings

from ..dispatcher import Dispatcher

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial

import typing
from typing import List

def gp_predictive_y(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood_var:np.ndarray, kernel: 'Kernel'):
    N = X.shape[0]

    K_xs = kernel.K(XS, XS)
    K_xs_x = kernel.K(XS, X)
    k_xx = kernel.K(X, X)

    k = k_xx + likelihood_var
    k_chol = cholesky(k) 

    mu = K_xs_x @ cholesky_solve(k_chol, Y)
    sig = K_xs - K_xs_x @ cholesky_solve(k_chol, K_xs_x.T)

    return mu, sig

def gp_predictive_y_diagonal(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood_var:np.ndarray, kernel: 'Kernel'):
    N = X.shape[0]

    K_xs = kernel.K_diag(XS)
    K_xs_x = kernel.K(XS, X)
    k_xx = kernel.K(X, X)

    k = k_xx + likelihood_var
    k_chol = cholesky(k) 

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y)
    sig = K_xs - np.sum(np.square(A1), axis=0)


    return mu, sig


@Dispatcher.register('predictors', GaussianLikelihood, diagonal=False)
def gp_predict_y(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood: GaussianLikelihood, prior: Distribution):
    kernel = prior.kernel
    N = X.shape[0]
    lik_var = likelihood.variance * np.eye(N)

    return gp_predictive_y(XS, X, Y, lik_var, kernel)

@Dispatcher.register('predictors', DiagonalGaussianLikelihood, diagonal=False)
def gp_predict_y_diag(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood: GaussianLikelihood, prior: Distribution):
    kernel = prior.kernel
    N = X.shape[0]
    lik_noise = np.diag(likelihood.variance)

    return gp_predictive_y(XS, X, Y, lik_var, kernel)


@Dispatcher.register('predictors', GaussianLikelihood, diagonal=True)
def gp_predict_y_diagional(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood:GaussianLikelihood, prior: KernelGaussianDistribution):
    kernel = prior.kernel
    N = X.shape[0]

    lik_var = likelihood.variance * np.eye(N)
    return gp_predictive_y_diagonal(XS, X, Y, lik_var, kernel)

@Dispatcher.register('predictors', DiagonalGaussianLikelihood, diagonal=True)
def diagonal_gp_predict_y_diagional(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood:GaussianLikelihood, prior: KernelGaussianDistribution):
    kernel = prior.kernel
    N = X.shape[0]

    lik_var = np.diag(likelihood.variance) 

    return gp_predictive_y_diagonal(XS, X, Y, lik_var, kernel)

@Dispatcher.register('predictors', NaturalDiagonalGaussianLikelihood, diagonal=True)
def natural_diagonal_gp_predict_y_diagional(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood:NaturalDiagonalGaussianLikelihood, prior: KernelGaussianDistribution):
    kernel = prior.kernel
    N = X.shape[0]

    Y = likelihood.Y
    lik_var = np.diag(np.squeeze(likelihood.variance))

    mu, var =  gp_predictive_y_diagonal(XS, X, Y, lik_var, kernel)

    return mu, var

@Dispatcher.register('predictors', NaturalBlockDiagonalGaussianLikelihood, diagonal=True)
def natural_block_diagonal_gp_predict_y_diagional(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood:NaturalDiagonalGaussianLikelihood, prior: KernelGaussianDistribution):
    kernel = prior.kernel
    N = X.shape[0]

    Y = likelihood.Y
    Y = np.reshape(Y, [Y.shape[0]*Y.shape[1], 1])
    lik_var = jax.scipy.linalg.block_diag(*likelihood.variance)

    mu, var =  gp_predictive_y_diagonal(XS, X, Y, lik_var, kernel)

    return mu, var

@Dispatcher.register('predictors', NaturalBlockDiagonalGaussianLikelihood, diagonal=False)
def natural_block_diagonal_gp_predict_y(XS: np.ndarray, X: np.ndarray, Y:np.ndarray, likelihood:NaturalDiagonalGaussianLikelihood, prior: KernelGaussianDistribution):
    kernel = prior.kernel
    N = X.shape[0]

    Y = likelihood.Y
    Y = np.reshape(Y, [Y.shape[0]*Y.shape[1], 1])
    lik_var = jax.scipy.linalg.block_diag(*likelihood.variance)

    mu, var =  gp_predictive_y(XS, X, Y, lik_var, kernel)

    return mu, var

def lmc_predict_y_diagional(XS:np.ndarray, X_arr: List[np.ndarray], Y_arr: List[np.ndarray], likelihood: LMC_Likelihood, prior_arr: List[KernelGaussianDistribution]) -> np.ndarray:
    raise NotImplementedError()
    num_latents = likelihood.num_latents
    num_outputs = likelihood.num_outputs

    XS_arr = [XS for i in range(num_outputs)] 

    K_x_x_prior = get_reparameterised_lmc_prior(X_arr, X_arr, likelihood, prior_arr)
    K_xs_x_prior = get_reparameterised_lmc_prior(XS_arr, X_arr, likelihood, prior_arr)
    K_xs_xs_prior = get_reparameterised_lmc_prior(XS_arr, XS_arr, likelihood, prior_arr)

    #add likelihood_variance to k_xx
    variances = likelihood.variances()
    for output in range(num_outputs):
        N_output = X_arr[output].shape[0]
        K_x_x_prior[output] = K_x_x_prior[output] + np.eye(X_arr[output].shape[0])*variances[output]

    #create block diagional matrices
    Y_vec = np.vstack(Y_arr) #NPx1

    k_x_x = jax.scipy.linalg.block_diag(*K_x_x_prior) #NPxNP
    k_xs_x = jax.scipy.linalg.block_diag(*K_xs_x_prior) #(NS*P)xNP
    k_xs_xs = jax.scipy.linalg.block_diag(*K_xs_xs_prior) #(NS*P)xNP
    k_chol = cholesky(k_x_x) 

    mu = k_xs_x @ cholesky_solve(k_chol, Y_vec)
    sig = k_xs_xs - k_xs_x @ cholesky_solve(k_chol, k_xs_x.T)

    return mu, sig

def lmc_predict_y(XS:np.ndarray, data:ListData, likelihood: LMC_Likelihood, prior_arr: List[KernelGaussianDistribution]) -> np.ndarray:
    """
         The LMC Model is defined as:

            X_arr = [X_p] where X_p ∈ N x D
            Y_arr = [Y_p] where Y_p ∈ N x 1
            
            Y = vec(Y_arr)
            X = vec(X_arr)

        with prior:

            p(F) = N(F| 0, (W ⊗ I) blkdiag(K_1(X, X), ..., K_P(X, X)) (W ⊗ I)^T) = N(F | K)

        The LMC Likelihood is:

            p(Y|F) = N(Y | F, diag(sigma_1, ..., sigma_P)) 

        The Predictive distribution follows directly from the GP predictive equations:

            mu* =  K(XS, X) [  K(X, X) + diag(sigma_1, ..., sigma_P)]^{-1} Y

            var* =  K(XS, XS) - K(XS, X) [  K(X, X) + diag(sigma_1, ..., sigma_P)]^{-1} K(X, XS)
    """

    X_arr = data.X
    Y_arr = data.Y

    num_latents = likelihood.num_latents
    num_outputs = likelihood.num_outputs

    mixing_weights = likelihood.coregion_weights

    XS_arr = [XS for i in range(num_outputs)] 

    #get prior of [X, XS]
    X_stacked = [np.vstack([X_arr[0], XS_arr[0]]) for q in range(num_latents)]

    #[N+NS]Q x [N+NS]Q
    K_stacked = get_blocks_from_lmc_proir(X_stacked, X_stacked, likelihood, prior_arr)

    W = np.kron(mixing_weights, np.eye(X_arr[0].shape[0]+XS_arr[0].shape[0]))
    variances = likelihood.variances()

    #reparameterised kernel [N+NS]P x [N+NS]P
    K_stacked = W @ K_stacked @ W.T

    N = X_arr[0].shape[0]
    Ns = XS.shape[0]
    augmented_size = X_arr[0].shape[0]+XS.shape[0]

    #NP x NP
    augmented_prior_kernel_x_x = np.zeros([N*num_outputs, N*num_outputs])
    for q in range(num_outputs):
        all_index_q = jax.ops.index[augmented_size*q:augmented_size*q+N, augmented_size*q:augmented_size*q+N]
        index_q = jax.ops.index[N*q:N*q+N, N*q:N*q+N]

        augmented_prior_kernel_x_x = jax.ops.index_add(augmented_prior_kernel_x_x, index_q, K_stacked[all_index_q])

    #NP x NsP
    augmented_prior_kernel_x_xs = np.zeros([N*num_outputs, Ns*num_outputs])
    for q in range(num_outputs):
        all_index_q = jax.ops.index[augmented_size*q:augmented_size*q+N, augmented_size*q+N:augmented_size*q+N+Ns]

        index_q = jax.ops.index[N*q:N*q+N, Ns*q:Ns*q+Ns]

        augmented_prior_kernel_x_xs = jax.ops.index_add(augmented_prior_kernel_x_xs, index_q, K_stacked[all_index_q])

    #NsP x NsP
    augmented_prior_kernel_xs_xs = np.zeros([Ns*num_outputs, Ns*num_outputs])
    for q in range(num_outputs):
        all_index_q = jax.ops.index[augmented_size*q+N:augmented_size*q+N+Ns, augmented_size*q+N:augmented_size*q+N+Ns]

        index_q = jax.ops.index[Ns*q:Ns*q+Ns, Ns*q:Ns*q+Ns]

        augmented_prior_kernel_xs_xs = jax.ops.index_add(augmented_prior_kernel_xs_xs, index_q, K_stacked[all_index_q])

    Y_vec = np.vstack(Y_arr) #NPx1

    variance_arr = [variances[output]*np.eye(X_arr[output].shape[0]) for output in range(num_outputs)]
    variance_arr = jax.scipy.linalg.block_diag(*variance_arr)

    #remove missing obsevations

    if True:
        bool_mask = np.logical_not(np.hstack(data.mask))
        Y_vec = Y_vec[bool_mask, :]
        augmented_prior_kernel_x_x = augmented_prior_kernel_x_x[bool_mask, ...]
        augmented_prior_kernel_x_x = augmented_prior_kernel_x_x[..., bool_mask]

        variance_arr = variance_arr[bool_mask, ...]
        variance_arr = variance_arr[..., bool_mask]

        #we only need to remove rows from x, not xs
        augmented_prior_kernel_x_xs = augmented_prior_kernel_x_xs[bool_mask, ...]

    jit = np.eye(variance_arr.shape[0])*Settings.jitter

    mu =  augmented_prior_kernel_x_xs.T @ np.linalg.solve(augmented_prior_kernel_x_x + variance_arr + jit , Y_vec)
    sig =  augmented_prior_kernel_xs_xs - augmented_prior_kernel_x_xs.T @ np.linalg.solve(augmented_prior_kernel_x_x + variance_arr + jit, augmented_prior_kernel_x_xs)

    pred_variance_arr = [variances[output]*np.eye(Ns) for output in range(num_outputs)]
    pred_variance_arr = jax.scipy.linalg.block_diag(*pred_variance_arr)

    sig += pred_variance_arr

    return mu, sig

def full_structured_lmc_predict_y_diagional(XS:np.ndarray, prior_arr: List[Distribution], likelihood:LMC_Likelihood, sparsity_arr: List[Sparsity], q:ApproximatePosterior) -> np.ndarray:
    print('full_structured_lmc_predict_y_diagional')
    num_latents = likelihood.num_latents
    num_outputs = likelihood.num_outputs

    mixing_weights = likelihood.coregion_weights
    variance = likelihood.variance

    if type(prior_arr[0]) is WhitenedKernelGaussianDistribution:
        #the prior is whitened
        block_prior = BlockWhitenedGaussianDistribution(prior_arr)
    else:
        block_prior = BlockGaussianDistribution(prior_arr)

    #TODO: assume same inducing points for now
    latent_mu, latent_sig = q.distribution.predict_f(XS, block_prior, sparsity_arr[0], diagional_var=True, predict=True)

    #latent_sig = np.diagonal(latent_sig)
    latent_sig = np.reshape(latent_sig, [latent_sig.shape[0], 1])

    NS = XS.shape[0]
    latent_mu_arr = [latent_mu[NS*q: NS*(q+1), :] for q in range(num_latents)]
    latent_sig_arr = [latent_sig[NS*q: NS*(q+1), :] for q in range(num_latents)]

    mu_arr = []
    sig_arr = []
    for output in range(num_outputs):
        output_weights = mixing_weights[output, :]
        mu = np.sum([output_weights[q]*latent_mu_arr[q] for q in range(num_latents)], axis=0)
        sig = np.sum([(output_weights[q]**2)*latent_sig_arr[q] for q in range(num_latents)], axis=0)
        sig = sig+variance[output]

        sig = np.reshape(sig, [sig.shape[0], 1])

        mu_arr.append(mu)
        sig_arr.append(sig)

    return mu_arr, sig_arr


