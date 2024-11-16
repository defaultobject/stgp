import jax
import jax.numpy as np
from jax import jit
import chex
import objax
from batchjax import batch_or_loop

from ..transforms import Independent, LinearTransform
from ..utils.utils import get_batch_type
from .matrix_ops import batched_balance_matrix, balance_matrix
from .. import settings

from typing import List


def get_diagonal_gaussian_likelihood_variances(Y: np.ndarray, likelihood) -> np.ndarray:
    num_latents = Y.shape[1]
    N = Y.shape[0]

    def _compute_lik_variance(N, likelihood):
        return likelihood.variance * np.eye(N)

    var_arr = batch_or_loop(
        _compute_lik_variance,
        [ N, likelihood ],
        [ None, 0 ],
        num_latents,
        1,
        get_batch_type(likelihood)
    )

    var_arr = jax.scipy.linalg.block_diag(*var_arr)

    chex.assert_equal(var_arr.shape[0], N * num_latents)
    chex.assert_equal(var_arr.shape[1], N * num_latents)

    return var_arr

def get_vec_gaussian_likelihood_variances(Y: np.ndarray, likelihood) -> np.ndarray:
    num_latents = Y.shape[1]
    N = Y.shape[0]

    def _compute_lik_variance(N, likelihood):
        return likelihood.variance * np.ones(N)

    var_arr = batch_or_loop(
        _compute_lik_variance,
        [ N, likelihood ],
        [ None, 0 ],
        num_latents,
        1,
        get_batch_type(likelihood)
    )

    return var_arr

def get_ss_balance_transformation(A_k):
    d = balance_matrix(A_k, settings.balance_state_space_iters)

    d_inv = 1/d
    D = np.diag(d)
    D_inv = np.diag(d_inv)

    return D, D_inv

def get_batched_ss_balance_transformation(A_arr):
    d = batched_balance_matrix(A_arr, settings.balance_state_space_iters)

    D = jax.vmap(np.diag)(d)
    d_inv = 1/d
    D_inv = jax.vmap(np.diag)(d_inv)

    return D, D_inv

@jit
def _transform_ss_A_param(T, T_inv, A_k):
    return T_inv @ A_k @ T

@jit
def _transform_ss_H_param(T, T_inv, H_k):
    return H_k @ T

@jit
def _transform_ss_Q_param(T, T_inv, Q_k):
    return T_inv @ Q_k @ T_inv.T

@jit
def _transform_ss_params(T, T_inv, A_k, Q_k, H_k):
    A_k = _transform_ss_A_param(T, T_inv, A_k)
    Q_k = _transform_ss_Q_param(T, T_inv, Q_k)
    H_k =  _transform_ss_H_param(T, T_inv, H_k)
    return A_k, Q_k, H_k

@jit
def _transform_ss_init_params(T_0, T_inv_0, m_inf, P_inf):
    P_inf = T_inv_0 @ P_inf @ T_inv_0.T
    m_inf = T_inv_0 @ m_inf 
    return  m_inf, P_inf
