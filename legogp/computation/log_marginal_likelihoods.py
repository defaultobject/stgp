# Import Types
from ..data import Data
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian, GaussianParameterised, ProductLikelihood, GaussianProductLikelihood, BlockDiagonalGaussian
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian, log_gaussian_with_nans
from ..transforms import Independent, LinearTransform
from .model_ops import get_diagonal_gaussian_likelihood_variances
from .matrix_ops import vec_columns, stack_rows
from ..models import BatchGP
from ..utils import utils
from ..utils.utils import get_batch_type
from .permutations import data_order_to_output_order


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

def gaussian_log_marginal_likelihood(
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


@dispatch(Data, BatchGP, ProductLikelihood, LinearTransform)
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: ProductLikelihood, prior: LinearTransform
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    likelihood_arr = likelihood.likelihood_arr
    # Assume that are likelihoods are the same such that they can be batched over
    assert all([type(lik) == Gaussian for lik in likelihood_arr])

    N = Y.shape[0]

    Y_vec = vec_columns(Y)

    mean = prior.vec_mean(X)
    K_xx = prior.full_covar(X, X)
    lik_xx = get_diagonal_gaussian_likelihood_variances(Y, likelihood_arr)

    sigma = K_xx + lik_xx

    return log_gaussian_with_nans(Y_vec, mean, sigma) 


@dispatch(Data, BatchGP, ProductLikelihood, Independent)
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: ProductLikelihood, prior: Independent
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    
    X = data.X
    Y = data.Y

    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    # precompute prior covariance
    k_xx_arr = prior.covar(X, X)
    mean_arr = prior.mean(X) 

    # Ensure batched Y has rank 2
    Y = Y[..., None]

    likelihood_arr = likelihood.likelihood_arr
    lml_fn = gaussian_log_marginal_likelihood

    # Compute lml for each likelihood and prior
    lml_arr = batch_or_loop(
        lambda lml_fn, X, Y, lik, k, mean: lml_fn(X, Y, lik, k, mean),
        [ lml_fn, X, Y, likelihood_arr, k_xx_arr, mean_arr],
        [ None, None, 1, 0, 0, 0],
        dim = num_latents,
        out_dim = 1,
        batch_type = get_batch_type(likelihood_arr)
    )

    lml =  np.sum(lml_arr)

    return lml

@dispatch(Data, BatchGP, BlockDiagonalGaussian, LinearTransform)
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: BlockDiagonalGaussian, prior: LinearTransform
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    X = data.X
    Y = data.Y

    chex.assert_rank(Y, 2)

    Y_vec = vec_columns(Y)


    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    # precompute prior covariance
    k_xx_arr = prior.full_covar(X, X)
    mean_arr = prior.vec_mean(X) 

    likelihood_var = likelihood.full_variance

    # Permute so that the ordering between likelihood_var and Y is the same
    N = X.shape[0]
    NS = likelihood_var.shape[0]

    P = data_order_to_output_order(num_outputs, N)

    ordered_likelihood_var = P @ likelihood_var @ P.T

    return log_gaussian_with_nans(
        Y_vec,
        mean_arr,
        k_xx_arr + ordered_likelihood_var
    )

@dispatch(Data, BatchGP, BlockDiagonalGaussian, 'DataLatentPermutation')
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: BlockDiagonalGaussian, prior: 'DataLatentPermutation'
):

    X = data.X
    Y = data.Y

    # precompute prior covariance
    # X is latent-data order. prior will permute this so that the output is in data-latent order.
    k_xx_arr = prior.full_covar(X, X)
    mean_arr = prior.vec_mean(X) 

    # Y is ordered by data x latent. To ensure data-latent order, 
    #   we want stack rows (ie Y that correspond to the same data point
    #   are next to each other).
    Y_vec = stack_rows(Y)

    # Likelihood is defined in data-latent order
    likelihood_var = likelihood.full_variance

    return log_gaussian_with_nans(
        Y_vec,
        mean_arr,
        k_xx_arr + likelihood_var
    )


@dispatch(Data, BatchGP, 'BlockGaussianProductLikelihood', 'DataLatentPermutation')
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: BlockDiagonalGaussian, prior: 'DataLatentPermutation'
):

    X = data.X
    Y = data.Y
    # TODO: needs to generalise
    likelihood = likelihood.likelihood_arr[0]

    # precompute prior covariance
    k_xx_arr = prior.full_covar(X, X)
    mean_arr = prior.vec_mean(X) 

    # Ensure batched Y has rank 2
    Y = Y[..., None]
    Y_vec = vec_columns(Y)

    likelihood_var = jax.scipy.linalg.block_diag(*likelihood.variance)

    return log_gaussian_with_nans(
        Y_vec,
        mean_arr,
        k_xx_arr + likelihood_var
    )

