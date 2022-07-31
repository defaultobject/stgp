import jax
from jax import jit
from functools import partial
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

from .. import settings
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..sparsity import NoSparsity, Sparsity, FullSparsity
from .. import utils
from .matrix_ops import cholesky, triangular_solve, add_jitter, diagonal_from_cholesky, cholesky_solve, diagonal_from_cholesky, block_diagonal_from_cholesky

from .predictors.base_predictors import gaussian_prediction_blocks

@jit
def gaussian_spatial_conditional_diagional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs_diag, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Computes:
        mu =  Ksz Kzz⁻¹ m
        var =  Ktt * diag[ Kss - Ksz Kzz⁻¹ Kss] + diag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """

    Kxsxs_diag = np.squeeze(Kxsxs_diag)
    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    
    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A2 = (S_chol.T @ A1).T # N x M 

    mu = mean_xs + A1.T @ (m-mean_x) # N x 1
    sig = Ktt * (Kxsxs_diag - np.sum(np.square(A), axis=0)) + np.sum(np.square(A2), axis=1) #N x 1

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

@jit
def gaussian_spatial_conditional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Computes:
        mu =  Ksz Kzz⁻¹ m
        var =  Ktt * blkdiag[ Kss - Ksz Kzz⁻¹ Kss] + blkdiag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """

    N = Kxsxs.shape[0]
    M = m.shape[0]

    chex.assert_shape(Kxsxs, [N, N])
    chex.assert_shape(Kzz, [M, M])
    chex.assert_shape(Kxz, [N, M])
    chex.assert_shape(mean_x, [M, 1])
    chex.assert_shape(mean_xs, [N, 1])
    chex.assert_shape(m, [M, 1])
    chex.assert_shape(S_chol, [M, M])

    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    
    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N

    A2 = (S_chol.T @ A1).T # N x M 

    mu = mean_xs + A1.T @ (m-mean_x) # N x 1
    sig = Ktt*(Kxsxs - A.T @ A) + A2 @ A2.T #N x N

    #ensure correct shapes
    mu = np.reshape(mu, [N, 1])
    sig = np.reshape(sig, [N, N])

    return mu, sig

@jit
def gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs_diag, m, S_chol, mean_x, mean_xs) -> np.ndarray:
    """
    Let A = Kxz Kzz⁻¹ then
        
        N(f_s) = ∫ N(f_s | A (f - mean_x) + mean_s, Kxx - A Kzx) N(f | m, S) df
               = N(f_s | mean_s + A (m-mean_x), Kxx - A Kzx + A S A^T)
    """

    return gaussian_spatial_conditional_diagional(
        XS, X, Kzz, Kxz, Kxsxs_diag, 1, m, S_chol, mean_x, mean_xs
    )

@jit
def gaussian_conditional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, m, S_chol, mean_x, mean_xs) -> np.ndarray:
    """
    Let A = K(XS, X) K(X, X)^{-1} then
        
        N(f_s) = \int N(f_s | A (f - mean_x) + mean_s, K(XS, XS) - A K(X, XS)) N(f \mid m, S) df
               = N(f_s | mean_s + A (m-mean_x), K(XS, XS) - A K(X, XS) + A S A^T)
    """
    N = Kxsxs.shape[0]
    M = m.shape[0]

    chex.assert_shape(Kxsxs, [N, N])
    chex.assert_shape(Kzz, [M, M])
    chex.assert_shape(Kxz, [N, M])
    chex.assert_shape(mean_x, [M, 1])
    chex.assert_shape(mean_xs, [N, 1])
    chex.assert_shape(m, [M, 1])
    chex.assert_shape(S_chol, [M, M])

    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    
    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N

    A2 = (S_chol.T @ A1).T # N x M 

    mu = mean_xs + A1.T @ (m-mean_x) # N x 1
    sig = Kxsxs - A.T @ A + A2 @ A2.T #N x N

    #ensure correct shapes
    mu = np.reshape(mu, [N, 1])
    sig = np.reshape(sig, [N, N])

    return mu, sig

@partial(jit, static_argnums=(0, 1))
def gaussian_conditional_blocks(group_size, block_size, XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxx, m, S_chol, mean_x, mean_xs) -> np.ndarray:

    M = X.shape[1]
    N = XS.shape[0]
    Q = block_size

    chex.assert_shape(Kzz, [M*Q, M*Q])
    chex.assert_shape(Kxz, [N*Q, M*Q])
    chex.assert_shape(Kxx, [N, Q, Q])
    chex.assert_shape(m, [M*Q, 1])
    chex.assert_shape(S_chol, [M*Q, M*Q])

    # Add jitter to help the cholesy solve
    jit = np.eye(Kzz.shape[0])*settings.jitter

    # pred_mu = Kxz(Kzz+jit)^{-1}m
    # pred_var = Kxx - Kxz(Kzz+jit)^{-1}Kxz.T
    pred_mu, pred_var = gaussian_prediction_blocks(
        group_size, block_size, m,  Kxx, Kxz, Kzz, mean_x, mean_xs, jit
    )

    chex.assert_shape(pred_mu, [N, Q])
    chex.assert_shape(pred_var, [N, Q, Q])


    # Compute KxzKzz^{-1}SKzz^{-1}Kxz.T
    K_chol = cholesky(add_jitter(Kzz, settings.jitter))
    A = cholesky_solve(K_chol, Kxz.T)
    A2 = S_chol.T @ A 
    B = block_diagonal_from_cholesky(A2.T, block_size)

    mu = pred_mu
    var = pred_var + B

    return mu, var

@jit
def gaussian_conditional_covar(X1:np.ndarray, X2:np.ndarray, X: np.ndarray, Kzz, Kxz, Kzx, Kxsxs, m, S_chol) -> np.ndarray:
    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    A1 = cholesky_solve(k_zz_chol, Kzx)
    sig1 = Kxsxs - Kxz @ A1
    sig2 = Kxz @ cholesky_solve(k_zz_chol, S_chol) @ S_chol.T @ A1

    sig = sig1 + sig2

    return sig

@jit
def whitened_gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs_diag, m, S_chol) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]

    """


    k_zz = Kzz
    k_xz = Kxz
    k_xx_diag = Kxsxs_diag
    mu = m
    sig_chol = S_chol

    chex.assert_rank(k_zz, 2)
    chex.assert_rank(k_xz, 2)
    chex.assert_rank(k_xx_diag, 1)
    chex.assert_rank(mu, 2)
    chex.assert_rank(sig_chol, 2)

    k_zz_chol = cholesky(add_jitter(k_zz, settings.jitter))

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)
    A2 = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, sig_chol, lower=False)

    sig = k_xx_diag - np.sum(np.square(A1), axis=0) + np.sum(np.square(A2), axis=1)

    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

@jit
def whitened_gaussian_conditional_full(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, m, S_chol) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """
    k_zz = Kzz
    k_xz = Kxz
    k_xsxs = Kxsxs
    mu = m
    sig_chol = S_chol
    sig = sig_chol @ sig_chol.T

    k_zz_chol = cholesky(add_jitter(k_zz, settings.jitter))


    A = cholesky_solve(k_zz_chol, k_xz.T)
    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    sig = k_xsxs - k_xz @ A + A1.T @ sig @ A1

    return mu, sig

