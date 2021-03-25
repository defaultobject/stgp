from ..settings import jitter
from ..kernel import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..batching import Batched
from ..sparsity import NoSparsity
from .. import utils
from .matrix_ops import cholesky, triangular_solve, add_jitter

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
    k_zz_chol = cholesky(add_jitter(Kzz, jitter))
    
    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A2 = (S_chol.T @ A1).T # N x M 

    mu = A1.T @ m # N x 1
    sig = Kxsxs_diag - np.sum(np.square(A), axis=0) + np.sum(np.square(A2), axis=1) #N x 1

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig


@dispatch(object, GaussianApproximatePosterior, object, NoSparsity)
def diagonal_marginal(X, approximate_posterior, kernel, sparsity):
    return approximate_posterior.m, approximate_posterior.S_diag

@dispatch(object, object, GaussianApproximatePosterior, object, NoSparsity)
def diagonal_marginal(XS, X, approximate_posterior, kernel, sparsity):
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
