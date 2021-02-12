"""
    Implementations of closed form conditional expectations
"""

from .. import Likelihood
from .. import Distribution

from ..dispatcher import Dispatcher

from ..likelihoods import GaussianLikelihood
from ..approximate_posteriors import *
from ..distributions import *
from ..sparsity import *

from .general import log_chol_matrix_det, cholesky_solve, cholesky, triangular_solve
from ..settings import Settings

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial

import typing
from typing import List

#TODO: jit?

@Dispatcher.register('conditional', KernelGaussianDistribution, DiagonalNaturalGaussianDistribution, None, diagional_var=True)
@Dispatcher.register('conditional', KernelGaussianDistribution, GaussianDistribution, None, diagional_var=True)
def kernel_gaussian_gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, g1:KernelGaussianDistribution, g2:GaussianDistribution) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """

    k_zz = g1.covar(X, X)
    k_xz = g1.covar(XS, X)
    k_xsxs_diag = g1.covar_diag(XS)

    mu = g2.mean(X)
    sig = g2.covar(X, X)
    k_sig_chol = g2.covar_chol(X, X)

    k_xsxs_diag = np.squeeze(k_xsxs_diag)

    k_zz_chol = cholesky(k_zz+Settings.jitter*np.eye(X.shape[0]))
    #k_sig_chol = np.linalg.cholesky(sig+Settings.jitter*np.eye(mu.shape[0]))
    
    A = triangular_solve(k_zz_chol, k_xz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A2 = (k_sig_chol.T @ A1).T # N x M 

    #mu = k_xz @ cholesky_solve(k_zz_chol, mu) # N x 1
    mu = A1.T @ mu # N x 1

    sig = k_xsxs_diag - np.sum(np.square(A), axis=0) + np.sum(np.square(A2), axis=1) #N x 1

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

@Dispatcher.register('conditional', KernelGaussianDistribution, GaussianDistribution, None, diagional_var=False)
def kernel_gaussian_gaussian_conditional(XS:np.ndarray, X: np.ndarray, g1:KernelGaussianDistribution, g2:GaussianDistribution) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """
    k_zz = g1.covar(X, X)
    k_xz = g1.covar(XS, X)
    k_xsxs = g1.covar(XS, XS)
    mu = g2.mean(X)
    sig = g2.covar(X, X)

    k_zz_chol = cholesky(k_zz+Settings.jitter*np.eye(k_zz.shape[0]))

    A = cholesky_solve(k_zz_chol, k_xz.T)
    mu = k_xz @ cholesky_solve(k_zz_chol, mu)
    sig = k_xsxs - k_xz @ A + A.T @ sig @ A

    return mu, sig



@Dispatcher.register('conditional', BlockWhitenedGaussianDistribution, GaussianDistribution, None, diagional_var=True)
@Dispatcher.register('conditional', WhitenedKernelGaussianDistribution, GaussianDistribution, None, diagional_var=True)
def whitened_kernel_gaussian_gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, g1:WhitenedKernelGaussianDistribution, g2:GaussianDistribution) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """

    k_zz = g1.covar(X, X)
    k_xz = g1.covar(XS, X)
    k_xx_diag = g1.covar_diag(XS)
    mu = g2.mean(X)
    sig = g2.covar(X, X)
    k_xx_diag = np.squeeze(k_xx_diag)

    k_zz_chol = cholesky(k_zz+Settings.jitter*np.eye(mu.shape[0]))
    sig_chol = cholesky(sig+Settings.jitter*np.eye(mu.shape[0]))

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)
    A2 = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, sig_chol, lower=False)

    sig = k_xx_diag - np.sum(np.square(A1), axis=0) + np.sum(np.square(A2), axis=1)

    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

@Dispatcher.register('conditional', BlockWhitenedGaussianDistribution, GaussianDistribution, None, diagional_var=False)
@Dispatcher.register('conditional', WhitenedKernelGaussianDistribution, GaussianDistribution, None, diagional_var=False)
def whitened_kernel_gaussian_gaussian_conditional(XS:np.ndarray, X: np.ndarray, g1:WhitenedKernelGaussianDistribution, g2:GaussianDistribution) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f) from a whitened distribution
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """
    k_zz = g1.covar(X, X)
    k_xz = g1.covar(XS, X)
    k_xx = g1.covar(X, X)
    k_xsxs = g1.covar(XS, XS)
    mu = g2.mean(X)
    sig = g2.covar(X, X)

    k_zz_chol = cholesky(k_zz+Settings.jitter*np.eye(k_zz.shape[0]))

    A = cholesky_solve(k_zz_chol, k_xz.T)
    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    sig = k_xsxs - k_xz @ A + A1.T @ sig @ A1

    return mu, sig

@Dispatcher.register('conditional', BlockWhitenedGaussianDistribution, GaussianDistribution, NoSparsity, diagional_var=True, predict=False)
@Dispatcher.register('conditional', WhitenedKernelGaussianDistribution, GaussianDistribution, NoSparsity, diagional_var=True, predict=False)
def whitened_kernel_gaussian_diagional(XS: np.ndarray, X: np.ndarray, g1:WhitenedKernelGaussianDistribution, g2:GaussianDistribution) -> np.ndarray:
    k_zz = g1.covar(X, X)
    k_xz = g1.covar(XS, X)
    k_xx_diag = g1.covar_diag(XS)
    mu = g2.mean(X)
    sig = g2.covar(X, X)
    k_xx_diag = np.squeeze(k_xx_diag)

    k_zz_chol = cholesky(k_zz+Settings.jitter*np.eye(mu.shape[0]))
    sig_chol = cholesky(sig+Settings.jitter*np.eye(mu.shape[0]))


    mu = k_zz_chol @ mu
    sig = np.sum(np.square(k_zz_chol @ sig_chol), axis=1)

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

@Dispatcher.register('conditional', KernelGaussianDistribution, None, SpatialSparsity, False, None)
def spatial_batched_kernel_gaussian_conditional_full(XS_space: np.ndarray, Z_space: np.ndarray, q_mu: np.ndarray, q_var: np.ndarray, prior):
    kernel = prior.kernel

    active_dims = np.array(list(range(XS_space.shape[1])))[1:]

    K_xz = kernel.K(XS_space, Z_space, active_dims=active_dims)
    K_zz = kernel.K(Z_space, Z_space, active_dims=active_dims)
    K_xx = kernel.K(XS_space, XS_space, active_dims=active_dims)

    K_zz_chol = cholesky(K_zz + Settings.jitter*np.eye(K_zz.shape[0]))

    #calculates [I \kron Kxz Kzz^{-1}] 

    #Kxz Kzz^{-1}
    A = cholesky_solve(K_zz_chol, K_xz.T).T 

    def batch_gp_mean(a, y):
        return a @ y

    pred_m = jax.vmap(batch_gp_mean, (None, 0), (0))(A, q_mu)

    #calculate 
    #   K_zz_chol + [I \kron Kxz Kzz^{-1}] S [I \kron Kxz Kzz^{-1}].T

    #K_zz_chol
    #assuming that k_t is stationary

    gp_pred = (K_xx - K_xz @ cholesky_solve(K_zz_chol, K_xz.T))*kernel.variance[0]

    def batch_pred(gp_pred, a, s):
        return gp_pred + a @ s @ a.T

    pred_var =  jax.vmap(batch_pred, (None, None, 0), (0))(gp_pred, A, q_var)

    return pred_m, pred_var

def gaussian_gaussian_diagonal_conditional(k_xx_diag, k_xz, k_zz, q_mu, q_var, time_var):
    k_xx_diag = np.squeeze(k_xx_diag)

    k_zz_chol = cholesky(k_zz+Settings.jitter*np.eye(k_zz.shape[0]))
    q_var_chol = cholesky(q_var+Settings.jitter*np.eye(k_zz.shape[0]))
    #k_sig_chol = np.linalg.cholesky(sig+Settings.jitter*np.eye(mu.shape[0]))
    
    A = triangular_solve(k_zz_chol, k_xz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A2 = (q_var_chol.T @ A1).T # N x M 

    #mu = k_xz @ cholesky_solve(k_zz_chol, mu) # N x 1
    mu = A1.T @ q_mu # N x 1

    sig = time_var*(k_xx_diag - np.sum(np.square(A), axis=0)) + np.sum(np.square(A2), axis=1) #N x 1

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig


#@Dispatcher.register('conditional', KernelGaussianDistribution, None, SpatialSparsity, True, None)
def kronecker_spatial_batched_kernel_gaussian_conditional(data_xs: 'Data', Z_space: np.ndarray, q_mu: np.ndarray, q_var: np.ndarray, prior):
    XS_space = data_xs.spatial_locations

    kernel = prior.kernel

    active_dims = np.array(list(range(XS_space.shape[-1])))[1:]

    K_xz = kernel.K(XS_space, Z_space, active_dims=active_dims)
    K_zz = kernel.K(Z_space, Z_space, active_dims=active_dims)
    K_xx_diag = kernel.K_diag(XS_space, active_dims=active_dims)

    K_xx_diag = K_xx_diag

    q_mu = np.expand_dims(q_mu, -1)

    pred_m, pred_var = jax.vmap(gaussian_gaussian_diagonal_conditional, (None, None, None, 0, 0), (0, 0))(K_xx_diag, K_xz, K_zz, q_mu, q_var)

    return pred_m, pred_var

@Dispatcher.register('conditional', KernelGaussianDistribution, None, SpatialSparsity, True, None)
def spatial_batched_kernel_gaussian_conditional(data_xs: 'Data', Z_space: np.ndarray, q_mu: np.ndarray, q_var: np.ndarray, prior):
    #TODO: what happens if observations are not on a grid? do we require that we need the same number of spatial points at each point?

    XS_space = data_xs.X
    if type(XS_space) is list:
        XS_space = XS_space[0]

    Nt = XS_space.shape[0]
    Nt = XS_space.shape[1]
    D = XS_space.shape[2]

    #XS_space = np.reshape(XS_space, [-1, D])
    #q_mu = np.reshape(q_mu, [-1, 1])
    #q_var = np.reshape(q_var, [-1, 1])

    #XS_space in  Nt x Ns x d
    #Z_space in M x D

    kernel = prior.kernel

    active_dims = np.array(list(range(D)))[1:]

    K_xz_tmp = kernel.K(data_xs.spatial_locations, Z_space, active_dims=active_dims)

    K_xz = jax.vmap(lambda XS:  kernel.K(XS, Z_space, active_dims=active_dims), (0), (0))(XS_space)

    K_zz = kernel.K(Z_space, Z_space, active_dims=active_dims) # M x M
    K_xx_diag = jax.vmap(lambda XS:  kernel.K_diag(XS, active_dims=active_dims), (0), (0))(XS_space)

    #q_mu = np.expand_dims(q_mu, -1)
    K_xx_diag = np.expand_dims(K_xx_diag, -1)

    time_var = kernel.variance[0]

    pred_m, pred_var = jax.vmap(gaussian_gaussian_diagonal_conditional, (0, 0, None, 0, 0, None), (0, 0))(K_xx_diag, K_xz, K_zz, q_mu, q_var, time_var)

    pred_var = pred_var 

    return pred_m, pred_var

