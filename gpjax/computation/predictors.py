from ..settings import jitter
from ..kernel import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..batching import Batched
from .. import utils
from .matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

@jit
def gaussian_prediction(Y, K_xs, K_xs_x, K_xx, lik_var):
    Ns = K_xs.shape[0]
    N = Y.shape[0]

    k = K_xx + np.eye(N)*lik_var

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y)
    sig = K_xs - A1.T @ A1

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])

    return mu, sig

    

@dispatch(object, object, object, Gaussian, object)
def predict(XS, X, Y, likelihood, kernel):

    Ns = XS.shape[0]
    N = X.shape[0]

    K_xs = kernel.K(XS, XS)
    K_xs_x = kernel.K(XS, X)
    K_xx = kernel.K(X, X)

    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_equal(K_xs.shape, (XS.shape[0], XS.shape[0]))

    return gaussian_prediction(Y, K_xs, K_xs_x, K_xx, likelihood.variance)

@dispatch(object, object, object, Gaussian, object)
def predict_diagonal(XS, X, Y, likelihood, kernel):

    Ns = XS.shape[0]
    N = X.shape[0]

    K_xs = kernel.K_diag(XS)
    K_xs_x = kernel.K(XS, X)
    K_xx = kernel.K(X, X)

    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_equal(K_xs.shape, (XS.shape[0], ))

    k = K_xx + np.eye(N)*likelihood.variance
    #k_chol = cholesky(add_jitter(k, jitter)) 

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y)
    sig = K_xs - np.sum(np.square(A1), axis=0)

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, 1])

    chex.assert_equal(mu.shape, (XS.shape[0], 1))
    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig



@dispatch(object, object, GaussianApproximatePosterior, Gaussian, object, object)
def predict_diagonal(XS, X,  approximate_posterior, likelihood, kernel, sparsity):
    m, S_diag = approximate_posterior.predictive_marginal(XS, X, kernel, sparsity)
    return m, S_diag + likelihood.variance

