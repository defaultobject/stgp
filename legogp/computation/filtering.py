import jax
from jax.ops import index, index_update, index_add
from jax.experimental import loops
from jax import jit
import jax.numpy as np
import chex

from ..settings import jitter
from .matrix_ops import cholesky, cholesky_solve, add_jitter
from .gaussian import log_gaussian, log_gaussian_with_mask

@jit
def kalman_step(Y_k, A_k, H_k, m_k, P_k, Q_k, R_k, mask_k):
    # -- KALMAN PREDICT --
    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k



    M = np.multiply(
        np.tile(H_k @ mask_k, [1, Y_k.shape[0]]),
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
    #K_xs_x = (H_k @ P_).T

    # Computes
    m_k = m_ + K @ v
    #m_k = gaussian_posterior_mean_update_with_nans(
    #        m_, K, Y_k, mu, np.squeeze(H_k @ mask_k)
    #)

    # Computes
    P_k = P_ - K @ S @ K.T
    #P_k = gaussian_posterior_variance_update_with_nans(
    #        P_, K_xs_x, S, K_xs_x.T, np.squeeze(H_k @ mask_k), mask_k
    #)

    #log marginal likelihood (assuming Gaussian likelihood)
    log_Z_k = np.sum(
        log_gaussian_with_mask(Y_k, mu, S, np.squeeze(H_k @ mask_k))
    )

    # mask is zero if nan, one if not
    # when mask is one we want to use the kalman update else use the kalman prediction

    #m_k = m_k * mask_k + m_ * (1-mask_k)

    #mask_kp = mask_k @ mask_k.T 

    #P_k = P_k * mask_kp + P_ * (1-mask_kp)

    return m_k, P_k, log_Z_k

def sequential_kalman_filter(X, Y, kernel: 'Kernel', likelihood: 'Likelihood', N: int, store_intermediate: bool = False):
    chex.assert_rank(X, 3)
    chex.assert_equal(X.shape[0], N)
    chex.assert_rank(Y, 3)
    chex.assert_equal(Y.shape[0], N)

    x_t = X[:, 0, 0]
    X_s = X[0, :, :]

    F, L, Qc, H, P_inf = kernel.to_ss(X_s)

    state_size = kernel.state_size()
    latent_size = P_inf.shape[0]

    m_inf = np.zeros([latent_size, 1])

    #dt = np.concatenate([np.array([0.0]), np.diff(x_t)])
    dt = np.diff(x_t)

    # Compute R

    # TODO: generalise to changing variance
    R = np.eye(X_s.shape[0])*likelihood.variance

    # nan masking
    # construct mask so we can track where nans are
    mask_y = np.squeeze((~np.isnan(Y)).astype(int))
    mask = np.repeat(mask_y, state_size, axis=1)[..., None]

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
            A_k = kernel.expm(dt_k, X_s)
            Q_k = P_inf - A_k @  P_inf @ A_k.T
            R_k = R

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

    return m, P, G

def sequential_rts_smoother(X, m_filtered, P_filtered, kernel: 'Kernel', likelihood: 'Likelihood', N: int):
    chex.assert_rank(X, 3)
    chex.assert_equal(X.shape[0], N)

    x_t = X[:, 0, 0]
    X_s = X[0, :, :]

    N_t = x_t.shape[0]
    N_s = X_s.shape[0]

    F, L, Qc, H, P_inf = kernel.to_ss(X_s)

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

            A_k = kernel.expm(dt_k, X_s)
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

        breakpoint()
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

    #m, P = filter_to_obvs(filtered_m, filtered_P, kernel)

    #return log_marginal_lik, m, P

    smoothed_m, smoothed_P = sequential_rts_smoother(
        X, filtered_m, filtered_P,  kernel, likelihood, N = N
    )


    return log_marginal_lik, smoothed_m, smoothed_P


