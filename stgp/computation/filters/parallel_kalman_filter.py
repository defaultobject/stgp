""" 
Closely following 
    https://github.com/EEA-sensors/parallel-gps/blob/main/pssgp/kalman/parallel.py
    https://github.com/AaltoML/BayesNewton/blob/61cb0ebb23afb12de0008882bc3d16b864b7149e/bayesnewton/ops.py#L521
"""
import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan, associative_scan

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, mat_inv
from ..gaussian import log_gaussian, log_gaussian_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

# Import types
from ...transforms.sdes import SDE, LTI_SDE

import objax
import chex

def _first_filtering_element(m, P, F, Q, H, R, y):
    m_ = F @ m
    P_ = F @ P @ F.T + Q

    S1 = H @ P_ @ H.T + R
    S1_chol = cholesky(add_jitter(S1, settings.jitter))
    K = cholesky_solve(S1_chol, H @ P_).T

    A = np.zeros_like(F)
    b = m_ + K @ (y - H @ m_)
    C = P_ - K @ S1 @ K.T


    S = H @ Q @ H.T + R
    S_chol = cholesky(add_jitter(S, settings.jitter))

    eta = F.T @ H.T @ cholesky_solve(S_chol, y)
    J = F.T @ H.T @  cholesky_solve(S_chol, H @ F) 

    return A, b, C, J, eta

def _first_filtering_element_nan(m, P, F, Q, H, R, y):
     
    A = np.zeros_like(F)
    b = m
    C = P
    eta = np.zeros_like(m)
    J = np.zeros_like(F)

    return A, b, C, J, eta



def _generic_filtering_element(F, Q, H, R, y):
    I = np.eye(F.shape[0])

    S = H @ Q @ H.T + R
    S_chol = cholesky(add_jitter(S, settings.jitter))
    K = cholesky_solve(S_chol, H @ Q.T).T

    A = (I - K @ H) @ F
    b = K @ y
    C = (I - K @ H) @ Q
    eta = F.T @ H.T @ cholesky_solve(S_chol, y)
    J = F.T @ H.T @  cholesky_solve(S_chol, H @ F) 

    return A, b, C, J, eta

def _generic_filtering_element_nan(F, Q, H, R, y):
    A = F
    b = np.zeros([F.shape[1], 1])
    C = Q
    J = np.zeros_like(Q)
    eta = np.zeros_like(b)

    return A, b, C, J, eta


def filtering_operator(x1, x2):
    """ combine individual elements """
    A_i, b_i, C_i, J_i, eta_i = x1
    A_j, b_j, C_j, J_j, eta_j = x2

    N, D = A_i.shape
    I = np.eye(D)

    def fix_inv(A , B):
        """ compute (I + AB)^{-1} = [A(A^-1 + B)]^{-1} = [(A^-1 + B)]^{-1} A^{-1}"""
        A_inv = mat_inv(A)
        tmp = A_inv + B
        tmp_chol = cholesky(add_jitter(tmp, settings.jitter))
        inv_tmp = cholesky_solve(tmp_chol, A_inv)
        return inv_tmp

    inv_tmp = fix_inv(C_i, J_j)
    Aj_tmp = A_j @ inv_tmp

    A = Aj_tmp @ A_i
    C = Aj_tmp @ C_i @ A_j.T + C_j
    b = Aj_tmp @ (b_i + C_i @ eta_j)+b_j

    inv_tmp = fix_inv(J_j, C_i)
    eta = A_i.T @ inv_tmp @ (eta_j - J_j @ b_i) + eta_i
    J = A_i.T @ inv_tmp @ J_j @ A_i + J_i

    # FORCE PSD
    # required to fix very small errors that propogate
    C = 0.5 * (C + C.T)
    J = 0.5 * (J + J.T)
    return A, b, C, J, eta

def make_filtering_elements():
    pass

@dispatch('parallel')
def filter(data, prior, R, Y, X_t, X_s, dt):
    # compute steady states
    P_inf = prior.P_inf(None, X_s, None)
    m_inf = prior.m_inf(None, X_s, None)
    H = prior.H(None, X_s, None)

    # precompute all filtering parameters
    # TODO: this is O(N_t)!! need to distribute
    #dt = np.ones(Y.shape[0])*dt[1]
    R_arr = R
    A_arr = jax.vmap(lambda dt_k: prior.expm(X_s, dt_k))(dt)
    Q_arr = jax.vmap(lambda A_k: P_inf - A_k @ P_inf @ A_k.T)(A_arr)
    H_arr = np.tile(H[None, ...], [R.shape[0], 1, 1])
    #Q_arr = Q_arr.at[0].set(P_inf)

    mask = get_same_shape_mask(Y)
    # collapse mask
    mask = np.any(np.any(mask, axis=1), axis=1).astype(int)

    Y = np.nan_to_num(Y)

    x_0 = _first_filtering_element(m_inf, P_inf, A_arr[0], P_inf, H, R[0], Y[0])
    x_0_nan = _first_filtering_element_nan(m_inf, P_inf, A_arr[0], P_inf, H, R[0], Y[0])

    # combine nan and observered
    x_0 = [
        x_0[i] * mask[0] + x_0_nan[i] * (1-mask[0])
        for i in range(5)
    ]

    x_all = jax.vmap(
        _generic_filtering_element
    )(A_arr, Q_arr, H_arr, R, Y)

    x_all_nan = jax.vmap(
        _generic_filtering_element_nan
    )(A_arr, Q_arr, H_arr, R, Y)

    # combine nan and observered
    def get_mask(mask, r):
        return np.reshape(mask, [-1] + [1]*(len(r.shape)-1))

    x_all = [
        x_all[i] * get_mask(mask, x_all[i]) + x_all_nan[i] * (1-get_mask(mask, x_all[i]))
        for i in range(5)
    ]

    x_all = [
        x_all[i].at[0].set(x_0[i])
        for i in range(5)
    ]

    res = associative_scan(
        jax.vmap(filtering_operator), 
        x_all
    )

    filtered_means = np.vstack([m_inf[None, ...], res[1][:-1]])
    filtered_cov = np.vstack([P_inf[None, ...], res[2][:-1]])

    obs_means = jax.vmap(
        lambda H_k, m_k, F_k: H_k @ F_k @ m_k
    ) (
        H_arr, filtered_means, A_arr
    ) 
    obs_covs = jax.vmap(
        lambda H_k, P_k, R_k, F_k, Q_k: H_k @ F_k @ P_k @ F_k.T @ H_k.T  + H_k @ Q_k @ H_k.T + R_k
    )(H_arr, filtered_cov, R_arr, A_arr, Q_arr)


    log_Z_k = jax.vmap(
        lambda Y_k, mu_k, S_k: np.sum(
            log_gaussian_with_mask(np.nan_to_num(Y_k), mu_k, S_k, get_same_shape_mask(Y_k)[:, 0])
        )
    ) (Y, obs_means, obs_covs)
        
    log_Z = np.sum(log_Z_k)

    return log_Z, {'m': res[1], 'P': res[2]}

