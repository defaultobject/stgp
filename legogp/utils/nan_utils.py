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
    Y = np.nan_to_num(Y, nan=0.0)
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

def mask_matrix(mask: np.ndarray) -> np.ndarray:
    return np.diag(mask)
