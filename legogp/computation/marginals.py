from .. import settings
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..batching import Batched
from ..sparsity import NoSparsity, Sparsity, FullSparsity
from .. import utils
from .matrix_ops import cholesky, triangular_solve, add_jitter, diagonal_from_cholesky, cholesky_solve, diagonal_from_cholesky

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

@jit
def gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs_diag, m, S_chol) -> np.ndarray:
    """
    tbd.
    """

    Kxsxs_diag = np.squeeze(Kxsxs_diag)
    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    
    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A2 = (S_chol.T @ A1).T # N x M 

    mu = A1.T @ m # N x 1
    sig = Kxsxs_diag - np.sum(np.square(A), axis=0) + np.sum(np.square(A2), axis=1) #N x 1

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

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

    print(k_zz)
    print('k_zz: ', np.sum(k_zz))
    print('k_zz_chol: ', np.sum(k_zz_chol), np.sum(cholesky(k_zz)), settings.jitter)

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)
    A2 = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, sig_chol, lower=False)

    sig = k_xx_diag - np.sum(np.square(A1), axis=0) + np.sum(np.square(A2), axis=1)

    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    print('mu: ', np.sum(mu))

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


@dispatch(NoSparsity)
def diagonal_marginal(X, mean, K_xx, K_xz, K_zz, m, S_chol, sparsity):
    return m, diagonal_from_cholesky(S_chol)


@dispatch(FullSparsity)
def diagonal_marginal(X, mean, K_xx, K_xz, K_zz, m, S_chol, sparsity):
    Z = sparsity.Z
    return gaussian_conditional_diagional(
        X, Z, K_zz, K_xz, K_xx, m, S_chol
    )



@dispatch(object, GaussianApproximatePosterior, object, NoSparsity)
def whitened_diagonal_marginal(X, approximate_posterior, kernel, sparsity):
    X = sparsity.X
    Kxx = kernel.K(X, X)
    K_chol = cholesky(add_jitter(Kxx, settings.jitter))

    m, S_chol = approximate_posterior.m, approximate_posterior.S_chol

    mu = K_chol @ m
    var = diagonal_from_cholesky(K_chol @ S_chol)

    return mu, var

@dispatch(object, GaussianApproximatePosterior, object, Sparsity)
def whitened_diagonal_marginal(X, approximate_posterior, kernel, sparsity):

    Z = sparsity.Z
    return whitened_diagonal_marginal(X, Z, approximate_posterior, kernel, sparsity)

@dispatch(object, object, GaussianApproximatePosterior, object, NoSparsity)
def _diagonal_marginal(XS, X, approximate_posterior, kernel, sparsity):
    X = sparsity.X


    Kxx = kernel.K(X, X)
    Kxsx = kernel.K(XS, X)
    Kxsxs_diag = kernel.K_diag(XS)
    m, S_chol = approximate_posterior.m, approximate_posterior.S_chol

    return gaussian_conditional_diagional(
        XS,
        X,
        Kxx,
        Kxsx,
        Kxsxs_diag,
        m, 
        S_chol
    )

@dispatch(object, object, GaussianApproximatePosterior, object, NoSparsity)
def whitened_diagonal_marginal(XS, X, approximate_posterior, kernel, sparsity):
    X = sparsity.X

    Kxx = kernel.K(X, X)
    Kxsx = kernel.K(XS, X)
    Kxsxs_diag = kernel.K_diag(XS)
    m, S_chol = approximate_posterior.m, approximate_posterior.S_chol

    return whitened_gaussian_conditional_diagional(
        XS,
        X,
        Kxx,
        Kxsx,
        Kxsxs_diag,
        m, 
        S_chol
    )

@dispatch(object, object, GaussianApproximatePosterior, object, Sparsity)
def whitened_diagonal_marginal(XS, X, approximate_posterior, kernel, sparsity):
    X = sparsity.Z

    Kxx = kernel.K(X, X)
    Kxsx = kernel.K(XS, X)
    Kxsxs_diag = kernel.K_diag(XS)
    m, S_chol = approximate_posterior.m, approximate_posterior.S_chol

    return whitened_gaussian_conditional_diagional(
        XS,
        X,
        Kxx,
        Kxsx,
        Kxsxs_diag,
        m, 
        S_chol
    )

@dispatch(object, object, GaussianApproximatePosterior, object, NoSparsity)
def whitened_full_marginal(XS, X, approximate_posterior, kernel, sparsity):
    X = sparsity.X

    Kxx = kernel.K(X, X)
    Kxsx = kernel.K(XS, X)
    Kxsxs = kernel.K(XS, XS)
    m, S_chol = approximate_posterior.m, approximate_posterior.S_chol

    return whitened_gaussian_conditional_full(
        XS,
        X,
        Kxx,
        Kxsx,
        Kxsxs,
        m, 
        S_chol
    )

@dispatch(object, object, GaussianApproximatePosterior, object, Sparsity)
def whitened_diagonal_marginal(XS, X, approximate_posterior, kernel, sparsity):
    X = sparsity.Z

    Kxx = kernel.K(X, X)
    Kxsx = kernel.K(XS, X)
    Kxsxs_diag = kernel.K_diag(XS)
    m, S_chol = approximate_posterior.m, approximate_posterior.S_chol

    return whitened_gaussian_conditional_diagional(
        XS,
        X,
        Kxx,
        Kxsx,
        Kxsxs_diag,
        m, 
        S_chol
    )

@dispatch(object, object, GaussianApproximatePosterior, object, Sparsity)
def whitened_full_marginal(XS, X, approximate_posterior, kernel, sparsity):
    X = sparsity.Z

    Kxx = kernel.K(X, X)
    Kxsx = kernel.K(XS, X)
    Kxsxs = kernel.K(XS, XS)
    m, S_chol = approximate_posterior.m, approximate_posterior.S_chol

    return whitened_gaussian_conditional_full(
        XS,
        X,
        Kxx,
        Kxsx,
        Kxsxs,
        m, 
        S_chol
    )
