import jax
from jax.ops import index, index_update, index_add
from jax.experimental import loops
from jax import jit
import jax.numpy as np
import chex

from ..settings import jitter
from .matrix_ops import cholesky, cholesky_solve, add_jitter
from .gaussian import log_gaussian

@jit
def kalman_step(Y_k, A_k, H_k, m_k, P_k, Q_k, R_k, mask_k):
    # -- KALMAN PREDICT --
    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k

    # -- KALMAN UPDATE --
    mu = H_k @ m_
    var = H_k @ P_ @ H_k.T

    #inovation mean and variance
    v = Y_k - mu
    S = var + R_k

    #Kalman Gain
    S = add_jitter(S, jitter)
    L = cholesky(S)
    K = (cholesky_solve(L, H_k @ P_)).T

    m_k = m_ + K @ v
    P_k = P_ - K @ S @ K.T

    #log marginal likelihood (assuming Gaussian likelihood)
    log_Z_k = np.sum(log_gaussian(Y_k, mu, S))

    # mask is zero if nan, one if not
    # when mask is one we want to use the kalman update else use the kalman prediction

    m_k = m_k * mask_k + m_ * (1-mask_k)
    P_k = P_k * mask_k + P_ * (1-mask_k)

    return m_k, P_k, log_Z_k

def sequential_kalman_filter(X, Y, kernel: 'Kernel', likelihood: 'Likelihood', N: int, store_intermediate: bool = False):
    chex.assert_rank(X, 3)
    chex.assert_equal(X.shape[0], N)
    chex.assert_rank(Y, 3)
    chex.assert_equal(Y.shape[0], N)

    x_t = X[:, 0, 0]
    X_s = X[0, :, :]

    F, L, Qc, H, P_inf = kernel.to_ss()

    state_size = P_inf.shape[0]

    m_inf = np.zeros([state_size, 1])

    dt = np.concatenate([np.array([0.0]), np.diff(x_t)])

    # Compute R
    R = np.ones(N)*likelihood.variance

    # nan masking
    
    # construct mask so we can track where nans are
    mask = np.squeeze((~np.isnan(Y)).astype(int))

    # replac nans with zero to avoid nans in code
    Y = np.nan_to_num(Y, nan=0.0)
    

    with loops.Scope() as s:
        s.log_marginal_lik = 0.0
        s.m, s.P = m_inf, P_inf
        s.filtered_mean = np.zeros([N, state_size, 1])
        s.filtered_cov = np.zeros([N, state_size, state_size])

        for k in s.range(N):
            Y_k = Y[k]
            dt_k = dt[k]
            A_k = kernel.expm(dt_k)
            Q_k = P_inf - A_k @  P_inf @ A_k.T
            R_k = R[k]

            m_k, P_k, log_marg_lik_k = kalman_step(
                Y_k, A_k, H, s.m, s.P, Q_k, R_k, mask[k]
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

    return m, P

def sequential_rts_smoother(X, m_filtered, P_filtered, kernel: 'Kernel', likelihood: 'Likelihood', N: int):
    chex.assert_rank(X, 3)
    chex.assert_equal(X.shape[0], N)

    x_t = X[:, 0, 0]
    X_s = X[0, :, :]

    F, L, Qc, H, P_inf = kernel.to_ss()

    state_size = P_inf.shape[0]

    dt = np.diff(x_t)
    #dt = np.concatenate([np.array([0.0]), np.diff(x_t)])

    with loops.Scope() as s:
        s.m, s.P = m_filtered[-1, ...], P_filtered[-1, ...]

        s.smoothed_mean = np.zeros([N, 1])
        s.smoothed_var = np.zeros([N, 1, 1])

        for k in s.range(N-2, -1, -1):
            dt_k = dt[k]

            A_k = kernel.expm(dt_k)
            Q_k = P_inf - A_k @  P_inf @ A_k.T

            m_filtered_k = m_filtered[k, ...]
            P_filtered_k = P_filtered[k, ...]

            H_k = H

            m, P = rts_smoother_step(
                m_filtered_k, P_filtered_k, s.m, s.P, A_k, Q_k
            )

            s.m = m
            s.P = P

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

def filter_to_obvs(filtered_m, filtered_P, kernel: 'Kernel'):
    _, _, _, H, _ = kernel.to_ss()
    m =  ((H @ filtered_m.T).T)[..., 0]
    P = ((H @ (filtered_P @ H.T).T).T)[..., 0]

    return m, P

def filter_and_smooth(X, Y, kernel: 'Kernel', likelihood: 'Likelihood', N: int):
    log_marginal_lik, filtered_m, filtered_P = sequential_kalman_filter(
            X, Y, kernel, likelihood, N = N, store_intermediate=True
    )

    m, P = filter_to_obvs(filtered_m, filtered_P, kernel)

    #return log_marginal_lik, m, P

    smoothed_m, smoothed_P = sequential_rts_smoother(
        X, filtered_m, filtered_P,  kernel, likelihood, N = N
    )


    return log_marginal_lik, smoothed_m, smoothed_P


