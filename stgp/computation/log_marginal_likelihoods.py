# Import Types
from ..data import Data, TransformedData
from ..kernels import Kernel, RBF
from ..likelihood import Likelihood, Gaussian, GaussianParameterised, ProductLikelihood, GaussianProductLikelihood, BlockDiagonalGaussian
from ..dispatch import dispatch, evoke
from ..utils.batch_utils import batch_over_module_types
from .gaussian import log_gaussian, log_gaussian_with_nans
from ..transforms import Independent, LinearTransform, Transform
from .model_ops import get_diagonal_gaussian_likelihood_variances
from .matrix_ops import vec_columns, stack_rows
from ..models import BatchGP
from ..utils import utils
from ..utils.utils import get_batch_type
from ..utils.nan_utils import get_same_shape_mask
from .permutations import data_order_to_output_order
from ..core.models import Model
from ..core.model_types import LinearModel, NonLinearModel


from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector
from ..utils.utils import can_batch

import jax
import jax.numpy as np
from jax import jit
import chex
from typing import List
import objax
from objax import ModuleList
from typing import List
from batchjax import batch_or_loop, BatchType

# =================================== Individual Likelihoods ===================================
@dispatch(Gaussian)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, likelihood: Gaussian, K: np.ndarray, mean: np.ndarray
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        log N(Y | 0, K(X, X) + lik.variance*I)

    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)

    N = X.shape[0]

    chex.assert_shape(Y, [N, 1])
    chex.assert_shape(mean, [N, 1])
    chex.assert_shape(K, [N, N])

    lik_noise = likelihood.full_variance

    k = K + lik_noise * np.eye(N)

    return log_gaussian_with_nans(Y, mean, k) 

@dispatch(BlockDiagonalGaussian)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, likelihood: BlockDiagonalGaussian, K: np.ndarray, mean: np.ndarray
):
    chex.assert_rank(Y, 3)
    chex.assert_rank(mean, 2)
    chex.assert_rank(K, 2)

    N, block_size, _ = Y.shape

    Y_vec = np.reshape(Y, [N * block_size, 1])
    lik_var = likelihood.full_variance

    chex.assert_shape(K, lik_var.shape)

    K = K + lik_var

    return log_gaussian_with_nans(Y_vec, mean, K) 

# ===============================================================================================
# ===============================================================================================
# ========================================  ENTRY POINTs ========================================
# ===============================================================================================
# ===============================================================================================


# ========================================= Independent =========================================
@dispatch(Data, BatchGP, ProductLikelihood, Independent)
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: ProductLikelihood, prior: Independent
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    X = data.X
    Y = data.Y

    num_outputs = prior.output_dim

    # precompute prior covariance
    k_xx_arr = prior.covar_blocks(X, X)
    mean_arr = prior.mean_blocks(X) 

    chex.assert_rank(k_xx_arr, 3)
    chex.assert_rank(mean_arr, 3)

    # Ensure batched Y has rank 2
    Y = Y[..., None]

    likelihood_arr = likelihood.likelihood_arr

    # Compute lml for each likelihood and prior
    lml_arr = batch_over_module_types(
        evoke_name = 'log_marginal_likelihood',
        evoke_params = [],
        module_arr = [likelihood_arr],
        fn_params = [X, Y, likelihood_arr, k_xx_arr, mean_arr],
        fn_axes = [None, 1, 0, 0, 0],
        dim = num_outputs,
        out_dim  = 1 
    )
    chex.assert_shape(lml_arr, (num_outputs, ))

    lml =  np.sum(lml_arr)

    return lml


# ====================================== Linear Transforms ======================================

@dispatch(Data, Model, Likelihood, LinearModel)
def log_marginal_likelihood( data, gp, likelihood, prior):
    #TODO: fix this
    breakpoint()


# ====================================== NonLinear Models ======================================
@dispatch(Data, Model, Likelihood, NonLinearModel)
def log_marginal_likelihood( data, gp, likelihood, prior):
    raise RuntimeError('Batch Inference is not supported for Nonlinear Models. Try using Variational inference instead.')
