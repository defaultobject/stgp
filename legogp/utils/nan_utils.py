"""
JIT requires that matrix sizes are known. Although masking can be done in advance, it is more convenient to allowing dyanmic masking. This has the side affect that a mask does not need to be passed around so that the resulting framework is cleaener.
"""

import jax
import jax.numpy as np
import objax
import chex

from ..computation.matrix_ops import cholesky, cholesky_solve, add_jitter
from ..settings import jitter

def get_mask(Y: np.ndarray) -> np.ndarray:
    """
    Returns 1 if Y_n is numeric, otherwise 0
    """
    return np.squeeze((~np.isnan(Y)).astype(int))

def mask_vector(Y, mask):
    # Element wise multiplication to make 'nan' element zero
    return Y * mask[:, None]

def mask_to_identity(K: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    K is (probably) a full p.s.d matrix. We want to decorrelate any ... tbd 
    """

    chex.assert_rank(K, 2)
    chex.assert_rank(mask, 1)
    chex.assert_equal(K.shape[0], mask.shape[0])
    chex.assert_equal(K.shape[1], mask.shape[0])

    N = K.shape[0]

    mask = np.tile(mask, [mask.shape[0], 1]) 

    K = K-np.eye(N)
    K = np.multiply(K, mask)
    K = np.multiply(K, mask.T)
    K = K+np.eye(N)

    return K

def gaussian_posterior_mean_update_with_nans(prior_m, K, Y, Y_mu, mask):
    Y = mask_vector(Y, mask)

    init_mean =  prior_m + K @ (Y-Y_mu)
    updated_mean = init_mean + K @ Y_mu

    return updated_mean
    #return init_mean

def gaussian_posterior_variance_update_with_nans(K_xs, K_xs_x, K_xx, K_x_xs, mask, mask_2):
    S = mask_to_identity(K_xx, mask)
    S_chol = cholesky(add_jitter(S, jitter))
    init_covar = K_xs - K_xs_x @ cholesky_solve(S_chol, K_x_xs)

    obs_mask = np.tile(mask_2, [1, K_xs_x.shape[1]])
    obs_mask_T = np.tile(mask, [K_xs_x.shape[0], 1])

    nan_mask = (1-obs_mask)
    nan_mask_T = (1-obs_mask_T)

    A = np.multiply(
        K_xs_x,
        nan_mask
    )

    A = np.multiply(
        K_xs_x,
        obs_mask_T
    )

    B = A.T

    C = np.multiply(
        K_xs_x,
        nan_mask
    )

    C = np.multiply(
        K_xs_x,
        nan_mask_T
    )

    updated_covar = init_covar +  A @ B + C @ B + (A @ B).T + C @ C.T

    return updated_covar
