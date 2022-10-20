"""
Dispatched Prediction Functions.

In general each function should return two items:

    - mu: rank 3: N x P x A
    - var: rank 4: N x P x A x A or N x 1 x PA x PA etc

where

    - N: number of data points
    - P: number of outputs
    - A: block size
"""
# Import Types
from ...data import Data
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian, GaussianParameterised, ProductLikelihood, DiagonalGaussian, Likelihood, BlockDiagonalGaussian
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior, ApproximatePosterior
from ...dispatch import dispatch, evoke
from ..gaussian import log_gaussian
from ...transforms import Independent, Transform, LinearTransform, NonLinearTransform, Aggregate
from ..permutations import data_order_to_output_order

from ...utils import utils
from ...utils.utils import can_batch, get_batch_type
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import mask_to_identity, get_mask, mask_vector

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns, get_block_diagonal, stack_rows
from ..model_ops import get_diagonal_gaussian_likelihood_variances

from .base_predictors import gaussian_prediction, gaussian_predictive_covar, gaussian_predictive_mean, gaussian_prediction_diagonal, gaussian_prediction_blocks

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList
from batchjax import batch_or_loop, BatchType


# =========================== Likelihood specific GPR prediction equations ===========================

@dispatch('BatchGP', BlockDiagonalGaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, ))
    chex.assert_rank(Y, 2)

    chex.assert_equal(Y.shape[1], likelihood.block_size)

    # Convert Gaussian likelihood noise to diagonal matrix
    lik_var = likelihood.full_variance

    Y_vec = vec_columns(Y)

    mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    # this function only supports one output, but for compatability add the extra dimensions
    mu = mu[..., None]
    var = var[..., None, None]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch('BatchGP', BlockDiagonalGaussian)
def predict_full(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, NS))
    chex.assert_rank(Y, 2)

    chex.assert_equal(Y.shape[1], likelihood.block_size)

    # Convert Gaussian likelihood noise to diagonal matrix
    lik_var = likelihood.full_variance

    Y_vec = vec_columns(Y)

    mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    # this function only supports one output, but for compatability add the extra dimensions
    mu = (mu.T)[None, ...]
    var = var[None, None, ...]

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

@dispatch('BatchGP', Gaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, ))
    chex.assert_rank(Y, 2)

    # Convert Gaussian likelihood noise to diagonal matrix
    lik_var = np.eye(N) * likelihood.variance

    mu, var = gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    # this function only supports one output, but for compatability add the extra dimensions
    mu = mu[..., None]
    var = var[..., None, None]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch('BatchGP', Gaussian)
def predict_covar(XS_1, XS_2, X, Y, likelihood, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs):
    lik_var = np.eye(K_xx.shape[0]) * likelihood.variance

    return  gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, lik_var)

@dispatch('BatchGP', BlockDiagonalGaussian)
def predict_covar(XS_1, XS_2, X, Y, likelihood, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs):
    lik_var = likelihood.full_variance

    return  gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, lik_var)

@dispatch(Data, 'BatchGP', ProductLikelihood, Independent)
def predict_covar(XS_1, XS_2, data, gp, likelihood, prior):
    X = data.X
    Y = data.Y

    num_latents = prior.output_dim
    num_outputs = prior.output_dim

    # precompute batched kernels
    
    K_xs = prior.covar_blocks(XS_1, XS_2)
    K_xx = prior.covar_blocks(X, X)
    K_xs_x = prior.covar_blocks(XS_1, X)
    K_x_xs = prior.covar_blocks(X, XS_2)
    mean_x = prior.mean_blocks(X)
    mean_xs_1 = prior.mean_blocks(XS_1)
    mean_xs_2 = prior.mean_blocks(XS_2)

    likelihood_arr = likelihood.likelihood_arr

    # Ensure Y is rank 2 after batching
    Y = Y[..., None]

    var_arr =  batch_over_module_types(
        'predict_covar',
        [gp],
        likelihood_arr,
        [XS_1, XS_2, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs_1],
        [None, None, None, 1, 0, 0, 0, 0, 0, 0, 0],
        num_latents,
        1
    )

    chex.assert_shape(var_arr, [num_outputs, XS_1.shape[0], XS_2.shape[0]])

    return var_arr


@dispatch(Data, 'BatchGP', ProductLikelihood, Independent)
def predict_blocks(XS, data, gp, likelihood, prior, block_size: int):
    """ 
    An Independent prior with a product likelihood are treated as separate models and batched over.
    """
    X = data.X
    Y = data.Y

    NS = XS.shape[0]
    N, P = Y.shape[0], Y.shape[1]
    num_outputs = prior.output_dim

    if block_size == 1:
        K_xs = prior.var_blocks(XS)
        # returns rank 3 but we need rank 2
        K_xs = K_xs[..., 0]
        evoke_name = 'predict_diagonal'
    elif block_size == NS:
        K_xs = prior.covar_blocks(XS, XS)
        evoke_name = 'predict_full'
    else:
        K_xs = prior.covar_blocks(XS, XS)
        evoke_name = 'predict_blocks'

    # Precompute batched kernels
    K_xx = prior.covar_blocks(X, X)
    K_xs_x = prior.covar_blocks(XS, X)
    mean_x = prior.mean_blocks(X)
    mean_xs = prior.mean_blocks(XS)

    likelihood_arr = likelihood.likelihood_arr

    # Ensure Y is rank 2 after batching
    if len(Y.shape) == 2:
        Y = Y[..., None]

    # Batch across all latents
    marginal_mu, marginal_var =  batch_over_module_types(
        evoke_name,
        [gp],
        likelihood_arr,
        [XS, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size],
        [None, None, 1, 0, 0, 0, 0, 0, 0, None],
        num_outputs,
        2
    )

    V_P, V_NS, _, V_B, _ = marginal_var.shape

    # fix shapes
    # each component will return rank (3, 4). But each component is only one ouput so we can remove that axis
    #   and reshape into the proper shape
    marginal_mu = marginal_mu[:, :, 0, ...]
    marginal_mu = np.transpose(marginal_mu, [1, 0, 2])
    chex.assert_shape(marginal_mu, [V_NS, num_outputs,  block_size])

    if V_NS == 1:
        # full prediction
        marginal_var = marginal_var[:, :, 0, ...]
        marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])
        chex.assert_shape(marginal_var, [1, num_outputs, NS, NS])
    else:
        marginal_var = marginal_var[..., 0]
        marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])
        # Mean field so we do not capture the correlations between Q
        chex.assert_shape(marginal_var, [NS, num_outputs, block_size, block_size])

    return marginal_mu, marginal_var


@dispatch(Data, 'BatchGP', ProductLikelihood, LinearTransform)
def predict_blocks(XS, data, gp, likelihood, prior, block_size: int):
    """ 
    A linear model is treated as a full joint model. To compute we stack Y and treat like a standard
       Gaussian
    """

    X = data.X
    Y = data.Y

    NS = XS.shape[0]
    N, P = Y.shape
    num_outputs = prior.output_dim

    # Computed stacked K and Y
    likelihood_arr = likelihood.likelihood_arr

    K_xx = prior.covar(X, X)
    K_xs_x = prior.covar(XS, X)
    mean_x = prior.mean(X)
    mean_xs = prior.mean(XS)

    Y_vec = vec_columns(Y)

    # TODO: generalise to different likelihoods
    lik_var = get_diagonal_gaussian_likelihood_variances(Y, likelihood_arr)

    if block_size == 1:
        K_xs = prior.var(XS)[..., 0]
        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

        # Convert from stacked and fix shapes
        mu = np.reshape(mu, [P, NS, 1])
        var = np.reshape(var, [P, NS, 1])
        mu = np.transpose(mu, [1, 0, 2])
        var = np.transpose(var, [1, 0, 2])[..., None]

    elif block_size == NS:
        # TODO: need to decide on a convention here, difference between returning 
        #   full
        #   blocks of size NS
        #   blocks of size P
        # Not all of this information can be stored in block_size
        #  Need a prediction type? or maybe use a string?
        K_xs = prior.covar(XS, XS)
        mu, var =  gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
        mu = np.reshape(mu, [P, NS, 1])
        var = var[None, None, ...]
    else:
        raise NotImplementedError()


    chex.assert_rank([mu, var], [3, 4])
    return mu, var

# =========================== Model Specific prediction equations ===========================
@dispatch(Data, 'BatchGP', ProductLikelihood, Independent)
@dispatch(Data, 'BatchGP', Likelihood, LinearTransform)
def predict(XS, data, gp, likelihood, prior, diagonal: bool):
    """ Only supports linear models.  """

    if diagonal:
        block_size = 1
    else:
        block_size = XS.shape[0]
        
    return evoke('predict_blocks', data, gp, likelihood, prior)(
        XS, data, gp, likelihood, prior, block_size
    )
    

# =========================== Entry Point For Variational Predictions ===========================

@dispatch(Likelihood, Transform, ApproximatePosterior, True)
@dispatch(Likelihood, Transform, ApproximatePosterior, False)
def predict(XS, data, likelihood, prior, approximate_posterior, inference, whiten, diagonal, **kwargs):
    return  evoke('marginal_prediction', approximate_posterior, likelihood, prior, whiten=whiten)(
        XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, **kwargs
    )
