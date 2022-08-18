"""
Dispatched functions for computing:
    1) q(u)
    2) q(f) = \int p(f | u) q(u) du

To make computing natural gradients easier we compute the ELL is computed by:
    1) Collecting appropriate paramters from the approximate posterior q(u):
        - (i.e the diagonal, block diagonal, full covariance, etc)
    2) Passing these to the appropiate marginal to compute q(f)
    3) Compute the ELL

This file contains the dispatched method for computing both q(u) and q(f)

The way marginals are computed is very general to support 
    - blocked likelihoods,
    - meanfield and full gaussian approximate posteriors,
    - linear / non linear transformations
    - differential operator transformations

With these a wide variety of variational GP based models can be constructed.
"""
import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part

from .linear_marginals import linear_marginal_blocks
# ================================== Dispatched q(f) ==============================

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    """ Catch all for single latent functions """
    N = q_m.shape[0]

    if out_block_dim == 1:
        q_S = diagonal_from_cholesky(q_S_chol)
    elif q_S_chol.shape[-1] < out_block_dim:
        raise RuntimeError()
    elif q_S_chol.shape[-1] > out_block_dim:
        # TODO: subsample
        pass

    # ensure correct shape
    q_S = np.reshape(q_S, [N, out_block_dim, out_block_dim])

    return q_m, q_S

# ================================== Dispatched q(f) ==============================

@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=True)
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim, whiten):
    latents_arr = prior.parent
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = data.X.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]
    out_block_arr = [out_block_dim for q in range(num_latents)]

    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.
    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal_blocks',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [data, q_m, q_S_chol, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
        fn_axes = [None, 0, 0, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2,
        evoke_kwargs = {'whiten': whiten}
    )

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, out_block_dim, out_block_dim])

    return marginal_mu, marginal_var

@dispatch(ApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim: int, whiten: bool):
    assert out_block_dim == 1

    sparsity_arr = prior.base_prior.get_sparsity_list()
    sparsity_type = sparsity_arr[0]

    fn = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, prior.parent, sparsity_type, whiten=whiten)

    # we must use the predictive distribution here 
    mean_fn = lambda XS: fn(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior.parent, sparsity_arr, out_block_dim, whiten)[0][0]

    # assume that XS1 == XS2
    var_fn = lambda XS1, XS2: fn(XS1, data, q_m, q_S_chol, approximate_posterior, likelihood, prior.parent, sparsity_arr, out_block_dim, whiten)[1][0]

    mu = prior.derivative_mean.mean_blocks_from_fn(data.X, mean_fn)
    var = jax.vmap(lambda x: prior.derivative_kernel.K_from_fn(x[None, ...], x[None, ...], var_fn))(data.X)

    mu = np.transpose(mu, [2, 1, 0])
    var = var[None, ...]

    return mu, var

@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim: int, whiten: bool):
    """ Recursively compute the transformed linear marginal. """

    return linear_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim, whiten, XS=None
    )

# list of linear priors
@dispatch(ApproximatePosterior, Likelihood, list, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, list, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim: int, whiten: bool):
    """ 
    Prior is a list of transformed to be comptued Recursively. Simple loop through and collect the results.

    Note we do not use batching and this method will only really be used when the transforms in the list are 
        different, and hence batching wont be applicable anyway.

    Recursively compute the transformed linear marginal.  
    """
    mu_list, var_list = [], []
    for p in prior:
        mu_p, var_p  = evoke('marginal_blocks', approximate_posterior, likelihood, p, whiten=whiten)(
            data, q_m, q_S_chol, approximate_posterior, likelihood, p, out_block_dim, whiten
        ) 

        mu_list.append(mu_p)
        var_list.append(var_p)

    return mu_list, var_list
    
# ===============================================================================================
# ===============================================================================================
# ========================================  ENTRY POINTs ========================================
# ===============================================================================================
# ===============================================================================================


# ============================ MEANFIELD APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # get output block size. NonLinear transforms are applied elementwise so only need the likelihood
    #   block size
    out_block_size = likelihood.block_size

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block_size, whiten
    ) 

    breakpoint()
    return val[0], val[1]



# ============================ FULL GAUSSIAN APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    out_block_size = max(
        likelihood.block_size,
        prior.base_prior.output_dim
    )

    breakpoint()

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block_size, whiten
    ) 
    breakpoint()
