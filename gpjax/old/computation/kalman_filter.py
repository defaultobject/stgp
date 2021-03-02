from ..dispatcher import Dispatcher
from ..likelihoods import (
    DiagonalGaussianLikelihood,
    GaussianLikelihood,
    BlockDiagonalGaussianLikelihood,
)

from ..settings import Settings

from ..computation.general import *
from ..computation.gaussian import *

import jax
import jax.numpy as jnp
from jax.ops import index, index_update, index_add
from jax.experimental import loops
from jax import jit, partial
import numpy as np

import pathlib
import json


# @partial(jit, static_argnums=(5, 6, 7, 8))
@jit
def kalman_step(y_k, A_k, H_k, m_k, P_k, Q_k, R_k, mask_k):
    # -- KALMAN PREDICT --
    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k

    # -- KALMAN UPDATE --
    mu = H_k @ m_
    var = H_k @ P_ @ H_k.T

    # inovation mean and variance
    v = y_k - mu

    if False:
        print("###############################################")
        print("R_k: ", R_k.shape)
        print("var: ", var.shape)
        print("y_k: ", y_k.shape)
        print("mu: ", mu.shape)
        print("H_k: ", H_k.shape)
        print("A_k: ", A_k.shape)

        print("R_k: ", np.sum(R_k))
        print("var: ", np.sum(var))
        print("mu: ", np.sum(m_k))
        print("P_: ", np.sum(P_))
        print("y_k: ", y_k)
        print("mu: ", np.sum(mu))
        print("H_k: ", H_k)
        print("A_k: ", A_k)

    S = var + R_k

    # Kalman Gain
    # This slows down the code a tiny bit, but not by enough to have to separate functions for the special case
    #   of K = P_ @ H.T / S
    S = S + Settings.jitter * np.eye(S.shape[0])
    L = cholesky(S)
    # print(L)
    K = (cholesky_solve(L, H_k @ P_)).T

    m_k = m_ + K @ v
    # P_k = P_ - K @ S @ K.T
    if False:
        # Joseph’s form covariance updatea
        I = np.eye(K.shape[0])
        A = I - K @ H_k
        P_k = A @ P_ @ A.T + K @ R_k @ K.T
    else:
        P_k = P_ - K @ S @ K.T

    # log marginal likelihood (assuming Gaussian likelihood)
    # log_Z_k = - jnp.log(jnp.maximum(2 * PI * (R_k + var), 1e-10)) / 2 - ((y_k - mu) ** 2) / (R_k + var) / 2
    log_Z_k = jnp.sum(log_gaussian(y_k, mu, S))

    # calculate choleskys and lin solve
    # alpha = v / S
    # chol_diag = jnp.sqrt(S)
    # beta = v / chol_diag

    alpha = 1.0
    chol_diag = 1.0
    beta = 1.0

    # mask
    # mask is True when y_k is nan

    # TODO: at the moment this will ignore the WHOLE spatial slice if there is any masks
    m_k = jnp.where(np.any(mask_k), m_, m_k)
    P_k = jnp.where(np.any(mask_k), P_, P_k)

    return m_k, P_k, log_Z_k, alpha, beta, chol_diag


# @partial(jit, static_argnums=(1, 2, 6))
# @partial(jit, static_argnums=(1))
def kalman_loop(y, N, dt, kernel, likelihood, sparsity, mask):
    F, L, Qc, H, P_inf = kernel.cf_to_ss(sparsity)

    latent_size = P_inf.shape[0]
    # zero mean GP
    m_inf = np.zeros([latent_size, 1])

    with loops.Scope() as s:
        s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
        s.m, s.P = m_inf, P_inf

        s.alpha, s.beta, s.chol_diag = (
            jnp.zeros_like(y[:, 0]),
            jnp.zeros_like(y[:, 0]),
            jnp.zeros_like(y[:, 0]),
        )

        s.sum_beta = 0.0

        for k in s.range(N):
            # for k in range(N):
            y_k = y[k]

            dt_k = dt[k]
            mask_k = mask[k]

            # TODO: Move outside
            A_k = kernel.expm(dt_k, sparsity)

            Q_k = P_inf - A_k @ P_inf @ A_k.T
            R_k = likelihood.get_variance_at_n(k, y_k.shape[0])

            H_k = H

            m_k, P_k, log_marg_lik_k, alpha, beta, chol_diag = kalman_step(
                y_k,
                A_k,
                H_k,
                s.m,
                s.P,
                Q_k,
                R_k,
                mask_k,
            )

            s.m = m_k
            s.P = P_k

            s.neg_log_marg_lik -= jnp.sum(log_marg_lik_k)

            # s.sum_beta += beta[0,0]**2

        return s.neg_log_marg_lik, None, None, s.alpha, s.sum_beta, s.chol_diag


# @partial(jit, static_argnums=(1, 2, 6))
# @partial(jit, static_argnums=(1))
def kalman_loop_store_intermediate(y, N, dt, kernel, likelihood, sparsity, mask):
    F, L, Qc, H, P_inf = kernel.cf_to_ss(sparsity)

    latent_size = P_inf.shape[0]
    # zero mean GP
    m_inf = np.zeros([latent_size, 1])

    with loops.Scope() as s:
        s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
        s.m, s.P = m_inf, P_inf

        s.filtered_mean = jnp.zeros([N, latent_size, 1])
        s.filtered_cov = jnp.zeros([N, latent_size, latent_size])

        s.alpha, s.beta, s.chol_diag = (
            jnp.zeros_like(y[:, 0]),
            jnp.zeros_like(y[:, 0]),
            jnp.zeros_like(y[:, 0]),
        )

        # TODO: hack to keep track of what index the training points vs testing points
        s.i = 0
        for k in s.range(N):
            # for k in range(N):
            y_k = y[k]
            dt_k = dt[k]
            mask_k = mask[k]

            A_k = kernel.expm(dt_k, sparsity)

            Q_k = P_inf - A_k @ P_inf @ A_k.T

            R_k = likelihood.get_variance_at_n(s.i, y_k.shape[0])
            m_k, P_k, log_marg_lik_k, alpha, beta, chol_diag = kalman_step(
                y_k,
                A_k,
                H,
                s.m,
                s.P,
                Q_k,
                R_k,
                mask_k,
            )

            s.m = m_k
            s.P = P_k

            # s.alpha = index_add(s.alpha, index[k], alpha[0][0])
            # s.beta = index_add(s.beta, index[k], beta[0][0])
            # s.chol_diag = index_add(s.chol_diag, index[k], chol_diag[0][0])

            s.i = jnp.where(np.any(mask_k), s.i, s.i + 1)

            s.filtered_mean = index_add(s.filtered_mean, index[k, ...], s.m)
            s.filtered_cov = index_add(s.filtered_cov, index[k, ...], s.P)

            s.neg_log_marg_lik -= jnp.sum(log_marg_lik_k)

        return (
            s.neg_log_marg_lik,
            s.filtered_mean,
            s.filtered_cov,
            s.alpha,
            s.beta,
            s.chol_diag,
        )


@Dispatcher.register("kalman_filter")
def kalman_loop_likelihood(
    y, N, dt, kernel, likelihood, sparsity, mask, store_intermediate
):

    if store_intermediate:
        return kalman_loop_store_intermediate(
            y, N, dt, kernel, likelihood, sparsity, mask
        )
    else:
        return kalman_loop(y, N, dt, kernel, likelihood, sparsity, mask)
