from ..dispatcher import Dispatcher
from ..likelihoods import DiagonalGaussianLikelihood, GaussianLikelihood

from .. import Settings

import jax.numpy as jnp
from jax.ops import index, index_update, index_add
from jax.scipy.linalg import cho_factor, cho_solve
from jax.experimental import loops
from jax import value_and_grad
from jax import random
from jax.nn import softplus
from jax import jit, partial
import numpy as np

import pathlib
import json

pi = 3.141592653589793
PI = 3.141592653589793

# @partial(jit, static_argnums=(1, 2, 8))
@partial(jit, static_argnums=(1))
def rts_smoother(
    y, N, dt, m_filtered, P_filtered, kernel, likelihood, sparsity, mask, alpha
):
    """
    Args:
        y: N x 1
        dt: N
        m_filtered: N x D x 1
        P_filtered:  N x D x D
    """

    F, L, Qc, H, P_inf = kernel.cf_to_ss(sparsity)

    latent_size = P_inf.shape[0]
    # zero mean GP
    m_inf = np.zeros([latent_size, 1])

    with loops.Scope() as s:
        s.m, s.P = m_filtered[-1, ...], P_filtered[-1, ...]

        s.smoothed_mean = jnp.zeros([N, y.shape[1]])
        s.smoothed_var = jnp.zeros([N, y.shape[1], y.shape[1]])
        s.alpha = alpha
        s.updated_alpha = jnp.zeros_like(s.alpha)

        s.A = kernel.expm(dt[-1], sparsity)

        for k in s.range(N - 2, -1, -1):
            # for k in range(N-2, -1, -1):
            y_k = y[k]
            # y_k = y[k*10:10*(k+1)]
            dt_k = dt[k]
            mask_k = mask[k]
            H_k = H

            # H_k = kernel.get_H(np.array([[0.0, 0.0]]))

            # TODO: Move outside
            A_k = s.A
            Q_k = P_inf - A_k @ P_inf @ A_k.T
            R_k = likelihood.get_variance_at_n(k, y_k.shape[0])

            m_predicted = A_k @ m_filtered[k, ...]
            P_predicted = A_k @ P_filtered[k, ...] @ A_k.T + Q_k

            # kalman gain
            P_predicted_chol, low = cho_factor(
                P_predicted + Settings.jitter * np.eye(P_predicted.shape[0])
            )
            G = cho_solve((P_predicted_chol, low), s.A @ P_filtered[k, ...]).T

            # update alpha to L^T / L / Y
            # a = s.alpha[k] - H_k @ G @ (s.m - s.A @ m_filtered[k, ...]) /  R_k

            s.A = kernel.expm(dt_k, sparsity)

            s.m = m_filtered[k, :] + G @ (s.m - m_predicted)
            s.P = P_filtered[k, :] + G @ (s.P - P_predicted) @ G.T

            # store the mean and var of f = Hx

            s.smoothed_mean = index_add(
                s.smoothed_mean, index[k, ...], np.squeeze((H_k @ s.m).T)
            )
            s.smoothed_var = index_add(
                s.smoothed_var,
                index[k, ...],
                jnp.squeeze(H_k @ s.P @ H_k.T),
            )

            if False:
                s.updated_alpha = index_add(s.updated_alpha, index[k], np.squeeze(a))

        if False:
            s.updated_alpha = index_add(s.updated_alpha, index[-1], s.alpha[-1])

        # H = kernel.get_H(np.array([[0.0, 0.0]]))
        s.smoothed_mean = index_add(
            s.smoothed_mean, index[-1, ...], jnp.squeeze((H @ m_filtered[-1, ...]).T)
        )
        s.smoothed_var = index_add(
            s.smoothed_var,
            index[-1, ...],
            jnp.squeeze(H @ P_filtered[-1, ...] @ H.T),
        )

    return s.smoothed_mean, s.smoothed_var, s.updated_alpha


@Dispatcher.register("rts_smoother", None)
def rts_smoother_likelihood(
    y, N, dt, m_filtered, P_filtered, kernel, likelihood, sparsity, mask, alpha
):

    return rts_smoother(
        y, N, dt, m_filtered, P_filtered, kernel, likelihood, sparsity, mask, alpha
    )
