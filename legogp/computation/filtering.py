import jax
from jax.ops import index, index_update, index_add
from jax.experimental import loops
from jax import jit
import jax.numpy as np
import chex

from ..settings import jitter
from .matrix_ops import cholesky, cholesky_solve, add_jitter
from .gaussian import log_gaussian, log_gaussian_with_mask
from ..utils.nan_utils import get_same_shape_mask

@jit
def kalman_step(Y_k, A_k, H_k, m_k, P_k, Q_k, R_k, mask_k):
    # -- KALMAN PREDICT --
    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k


    # Construct spatial mask
    M = np.multiply(
        np.tile(mask_k, [1, Y_k.shape[0]]),
        np.eye(Y_k.shape[0])
    )


    # -- KALMAN UPDATE --
    mu = M @ H_k @ m_
    var = M @ H_k @ P_ @ H_k.T @ M.T

    #inovation mean and variance
    v = Y_k - mu
    S = var + R_k

    #Kalman Gain
    S = add_jitter(S, jitter)
    L = cholesky(S)
    K = (cholesky_solve(L, M @ H_k @ P_)).T

    # Kalman Update
    m_k = m_ + K @ v
    P_k = P_ - K @ S @ K.T

    #log marginal likelihood (assuming Gaussian likelihood)
    log_Z_k = np.sum(
        log_gaussian_with_mask(Y_k, mu, S, mask_k[:, 0])
    )

    return m_k, P_k, log_Z_k

def sequential_kalman_filter(data: 'SequentialData', prior: 'Prior', likelihood: 'Likelihood', N: int, store_intermediate: bool = False):
    x_t = data.X_time
    X_s = data.X_space
    Y = data.Y

    # Get Filter parameters for corresponding prior
    F, L, Qc, H, P_inf  = prior.state_space_representation(X_s)

    # Dont chex X_s shape as it might be None in temporal setting
    chex.assert_rank(x_t, 1)
    chex.assert_rank(Y, 3)

    state_size = H.shape[0]
    latent_size = P_inf.shape[0]

    m_inf = np.zeros([latent_size, 1])

    dt = np.diff(x_t)

    # TODO: this wont work for the normal gaussian
    R = likelihood.variance

    # nan masking
    # construct mask so we can track where nans are
    mask_y = get_same_shape_mask(Y)

    # replace nans with zero to avoid nans in code
    Y = np.nan_to_num(Y, nan=0.0)

    with loops.Scope() as s:
        s.log_marginal_lik = 0.0
        s.m, s.P = m_inf, P_inf
        s.filtered_mean = np.zeros([N, latent_size, 1])
        s.filtered_cov = np.zeros([N, latent_size, latent_size])

        for k in s.range(N):
            Y_k = Y[k]
            dt_k = dt[k]
            A_k = prior.expm(dt_k, X_s)
            Q_k = P_inf - A_k @  P_inf @ A_k.T
            R_k = R[k]

            m_k, P_k, log_marg_lik_k = kalman_step(
                Y_k, A_k, H, s.m, s.P, Q_k, R_k, mask_y[k]
            )

            s.m = m_k
            s.P = P_k

            if store_intermediate:
                s.filtered_mean = index_add(s.filtered_mean, index[k, ...], s.m)
                s.filtered_cov = index_add(s.filtered_cov, index[k, ...], s.P)

            s.log_marginal_lik += np.sum(log_marg_lik_k)

        if store_intermediate:
            return s.log_marginal_lik, s.filtered_mean, s.filtered_cov
        else:
            return s.log_marginal_lik

@jit
def rts_smoother_step(m_filtered_k, P_filtered_k, m, P, A_k, Q_k):
    m_predicted = A_k @ m_filtered_k
    P_predicted = A_k @ P_filtered_k @ A_k.T + Q_k

    #kalman gain
    P_predicted_chol = cholesky(
        add_jitter(P_predicted, jitter)
    )
    G = cholesky_solve(
        P_predicted_chol, A_k @ P_filtered_k
    ).T

    m = m_filtered_k + G @ (m - m_predicted)
    P = P_filtered_k + G @ (P - P_predicted) @ G.T

    return m, P, G

def sequential_rts_smoother(data, m_filtered, P_filtered, prior: 'Prior', likelihood: 'Likelihood', N: int):
    x_t =  data.X_time
    X_s =  data.X_space

    N_t = data.Nt
    N_s = data.Ns

    F, L, Qc, H, P_inf  = prior.state_space_representation(X_s)

    state_size = P_inf.shape[0]

    dt = np.diff(x_t)

    with loops.Scope() as s:
        s.m, s.P = m_filtered[-1, ...], P_filtered[-1, ...]

        s.smoothed_mean = np.zeros([N_t, N_s])
        s.smoothed_var = np.zeros([N_t, N_s, N_s])
        s.Gs = np.zeros([N_t, state_size, state_size])
        s.Ps = np.zeros([N_t, state_size, state_size])

        for k in s.range(N-2, -1, -1):
            dt_k = dt[k]

            A_k = prior.expm(dt_k, X_s)
            Q_k = P_inf - A_k @  P_inf @ A_k.T

            m_filtered_k = m_filtered[k, ...]
            P_filtered_k = P_filtered[k, ...]

            H_k = H

            m, P, G = rts_smoother_step(
                m_filtered_k, P_filtered_k, s.m, s.P, A_k, Q_k
            )

            s.m = m
            s.P = P

            s.Gs = index_add(
                s.Gs, index[k, ...], G
            )

            s.Ps = index_add(
                s.Ps, index[k, ...], P
            )

            s.smoothed_mean = index_add(
                s.smoothed_mean, index[k, ...], np.squeeze((H_k @ s.m).T)
            )
            s.smoothed_var = index_add(
                s.smoothed_var,
                index[k, ...],
                np.squeeze(H_k @ s.P @ H_k.T)
            )

        s.smoothed_mean = index_add(
            s.smoothed_mean, index[-1, ...], np.squeeze((H @ m_filtered[-1, ...]).T)
        )
        s.smoothed_var = index_add(
            s.smoothed_var,
            index[-1, ...],
            np.squeeze(H @ P_filtered[-1, ...] @ H.T)
        )

        return s.smoothed_mean, s.smoothed_var

def filter_to_obvs(filtered_m, filtered_P, prior: 'Prior'):
    _, _, _, H, _ = prior.state_space_representation()
    m =  ((H @ filtered_m.T).T)[..., 0]
    P = ((H @ (filtered_P @ H.T).T).T)[..., 0]

    return m, P

def filter_and_smooth(data: 'SequentialData', prior: 'Prior', likelihood: 'Likelihood', N: int):
    log_marginal_lik, filtered_m, filtered_P = sequential_kalman_filter(
            data, prior, likelihood, N = N, store_intermediate=True
    )

    smoothed_m, smoothed_P = sequential_rts_smoother(
        data, filtered_m, filtered_P,  prior, likelihood, N = N
    )

    return log_marginal_lik, smoothed_m, smoothed_P


