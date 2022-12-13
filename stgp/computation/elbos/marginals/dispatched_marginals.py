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
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full, gaussian_conditional_blocks, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve, batched_block_diagional
from ...permutations import left_permute_mat, data_order_to_output_order, permute_vec, permute_mat, unpermute_vec, unpermute_mat
from ....core import Block, get_block_dim

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation, DataLatentPermutation
from ....transforms.latent_variable import LatentVariable
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateApproximatePosterior, FullConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior

from .linear_marginals import linear_marginal_blocks

# ================================== NoSparsity Entry Points ==============================
@dispatch(ConjugateApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
@dispatch(ConjugateApproximatePosterior, Likelihood, Transform, 'NoSparsity', whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, Transform, 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    With conjugate gaussian there is no need to convert from cholesky parameterizations.  
    
    q_m is in time-(space x)latent format. 
    """
    chex.assert_rank([q_m, q_S], [3, 4])

    N = q_m.shape[0]
    Q = prior.output_dim
    block_size = q_S.shape[-1]
    out_block_dim = get_block_dim(
        out_block, 
        approximate_posterior=approximate_posterior,
        likelihood = likelihood
    )

    if block_size == 1:
        q_m = np.reshape(q_m, [N, prior.output_dim, 1])
        q_S = np.reshape(q_S, [N, 1, prior.output_dim, prior.output_dim])

        return q_m, q_S

    if out_block_dim in [block_size, Q] :
        # only return the block diagonals across latents 
        # q_m and q_S are in time-latent-space format
        # to convert to data-latent format we first need convert each time point
        # to space-latent format, and then we can just reshape

        # convert to time-space-latent
        mu_p = jax.vmap(lambda a: permute_vec(a, Q))(q_m)
        var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(q_S)

        if out_block_dim == block_size:
            var_p = var_p[:, None, ...]
            return mu_p, var_p

        # extract block diagonals
        mu_p_bd = np.reshape(mu_p, [-1, Q, 1])
        var_p_bd = batched_block_diagional(var_p, Q)
        var_p_bd = np.reshape(var_p_bd, [-1, 1, Q, Q])

        chex.assert_rank([mu_p_bd, var_p_bd], [3, 4])
        return mu_p_bd, var_p_bd

    breakpoint()
    raise RuntimeError()

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    Catch all for single latent functions with no sparsity and no whitening.

    In general the variational params are stored in cholesky format so we only need to form the full covariance.
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    N = q_m.shape[0]

    out_block_dim = get_block_dim(out_block)

    if out_block_dim == 1:
        # assuming that q_m, q_S corresponds to a (single) full Gaussian
        # therefore the first two dimensions of q_S_chol are just 1
        chex.assert_equal([q_S_chol.shape[0], q_S_chol.shape[0]], [1, 1])
        q_S = diagonal_from_cholesky(q_S_chol[0, 0])

    elif q_S_chol.shape[-1] < out_block_dim:
        raise RuntimeError()

    elif q_S_chol.shape[-1] > out_block_dim:
        # TODO: subsample
        raise RuntimeError()

    # ensure correct shape
    
    q_m = np.reshape(q_m, [N, 1, out_block_dim])
    q_S = np.reshape(q_S, [N, 1, out_block_dim, out_block_dim])

    return q_m, q_S

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ Catch all for single latent functions with no sparsity but with whitening"""
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    N = q_m.shape[0]

    q_m = q_m[:, 0, ...]
    q_S_chol = q_S_chol[0, 0, ...]

    chex.assert_rank(q_S_chol, 2)
    chex.assert_shape(q_S_chol, [q_m.shape[0], q_m.shape[0]])

    Kzz = prior.covar(sparsity.Z, sparsity.Z)
    Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    out_block_dim = get_block_dim(out_block)

    if out_block_dim == 1:
        q_m = Kzz_chol @ q_m
        q_S = diagonal_from_cholesky(Kzz_chol @ q_S_chol)
    else:
        raise NotImplementedError()

    # ensure correct shape
    q_m = q_m[..., None]
    q_S = np.reshape(q_S, [N, 1, out_block_dim, out_block_dim])

    return q_m, q_S

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """
    The approximate posterior (and prior) is stored in latent-data format, this is because the prior is generally block diagional.
    However when computing the expected log likelihood, the likelihood decomposes across data points and hence we need in data-latent format.
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    q_m = q_m[:, 0, ...]
    q_S_chol = q_S_chol[0, 0, ...]

    assert isinstance(prior, DataLatentPermutation)

    out_block_dim = get_block_dim(
        out_block, 
        approximate_posterior=approximate_posterior
    )

    m = q_m
    S = q_S_chol @ q_S_chol.T

    Ns = m.shape[0]

    # Use the base prior as the transformation happens in thre ELL for Full posteriors
    num_latents = prior.base_prior.output_dim

    # X is shaped so that all outputs are grouped together
    # We need to instead group by each input

    m_p = prior.permute_vec(m, num_latents)
    S_p = prior.permute_mat(S, num_latents)

    m_p = np.reshape(m_p, [-1, num_latents])

    # Extract block diagonals
    S_blocks = get_block_diagonal(S_p, num_latents)

    m_p = m_p[..., None]
    S_blocks = S_blocks[:, None, ...]

    # Assert shapes are correct
    chex.assert_shape(m_p, [Ns/num_latents, num_latents, 1])
    chex.assert_shape(S_blocks, [Ns/num_latents, 1, num_latents, num_latents])

    return m_p, S_blocks

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    # first reparameterise and then we can treat as use

    # reparameterise
    base_prior = prior.base_prior

    Z = base_prior.get_Z_blocks()
    chex.assert_rank(Z, 3)

    Kzz = base_prior.b_covar(Z, Z)
    chex.assert_rank(Kzz, 2)

    Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    chex.assert_equal_shape([Kzz_chol, q_S_chol])

    q_m, q_S_chol =  Kzz_chol @ q_m, Kzz_chol @ q_S_chol

    # we have reparemeterised the approximate posterior so we can now treat it as unwhitened
    fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity[0], whiten=False)

    return fn(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block, False
    ) 


# ================================== Sparsity Entry Points ==============================
@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', Sparsity, whiten=False)
@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', Sparsity, whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    Catch all for single latent functions with sparsity.

    In general, sparsity can be considered as simply using the predictive distribution for q(f). 
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    M = q_m.shape[0]

    # TODO: block dim

    # Call the predictive distribution to compute q(f)
    mu, var =  evoke(
        'marginal_prediction_blocks', approximate_posterior, likelihood, 'GPPrior', sparsity, whiten=False 
    )(
        data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

#@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, Sparsity, whiten=False)
#@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, Sparsity, whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    In general, sparsity can be considered as simply using the predictive distribution for q(f). 
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    M = q_m.shape[0]

    # TODO: block dim

    # Call the predictive distribution to compute q(f)
    mu, var =  evoke(
        'marginal_prediction_blocks', approximate_posterior, likelihood, prior, sparsity[0], whiten=False 
    )(
        data, sparsity[0].raw_Z, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var


@dispatch(FullConjugateGaussian, Likelihood, Transform, Sparsity, whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, Transform, Sparsity, whiten=True)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    When using sparsity with a conjugate approximate posterior we handle it in the following way.

    When computing the ELBO we require the marginal q(f), to compute this we use the conjugate property to compute q(u) and then implement the conditionals. This is so we can take gradients through the integral.

    When predicting we can simply use the predictive distribution of the conjugate posterior.
    """
    breakpoint()
    # TODO: assuming that data_xs and data_x are of the same type
    chex.assert_rank([q_m, q_S], [3, 4])

    #parent is wrapped by a permutator, we don't need this so we pass the parent
    mu, var = evoke('spatial_conditional', data, prior.parent, approximate_posterior)(
        data, 
        sparsity[0].raw_Z, 
        q_m, 
        q_S[:, 0, ...], 
        approximate_posterior,
        likelihood,
        prior.parent,
        sparsity,
        out_block,
        whiten
    )

    #breakpoint()
    out_block_dim = get_block_dim(
        out_block, 
        approximate_posterior=approximate_posterior,
        likelihood = likelihood
    )

    # for testing
    #mu = q_m
    #var = q_S

    Q = prior.base_prior.output_dim
    block_size = var.shape[-1]

    # return either the full var, or the blocks across the latent functions
    if out_block_dim in [Q, block_size]: 
        # convert mu-var to data-latent format and extract block diagonal
        Q = prior.output_dim

        mu_p = jax.vmap(lambda a: permute_vec(a, Q))(mu)
        var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(var)

        if out_block_dim == block_size:
            var_p = var_p[:, None, ...]
            return mu_p, var_p

        mu_p = np.reshape(mu_p, [-1, Q, 1])
        var_p = batched_block_diagional(var_p, Q)
        var_p = np.reshape(var_p, [-1, 1, Q, Q])

        return mu_p, var_p
    else:
        breakpoint()
        raise NotImplementedError()


# ================================== Dispatched q(f) ==============================

@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.base_prior.get_sparsity_list()

    base_prior = get_permutated_prior(prior)

    fn = evoke('marginal_blocks', approximate_posterior, likelihood, base_prior, sparsity_arr[0], whiten=whiten)

    mu, var = fn(
        data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten
    ) 
    chex.assert_rank([mu, var], [3, 4])

    return mu, var



# Mean-field entry point
@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    latents_arr = prior.parent
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = data.X.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]

    # TODO: fix block sizes here
    out_block_arr = [likelihood_arr[0].block_type for q in range(num_latents)]

    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.
    # Compute q(f) for each output
    # add additional dimension to q_m and q_S_chol to ensure rank [3, 4] after batching
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal_blocks',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [data, q_m[:, :, None, ...], q_S_chol[:, :, None, ...], approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
        fn_axes = [None, 1, 1, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2,
        evoke_kwargs = {'whiten': whiten}
    )

    chex.assert_rank([marginal_mu, marginal_var], [4, 5])
    # fix shapes
    # each component will return rank (3, 4). But each component is only one ouput so we can remove that axis
    #   and reshape into the proper shape
    marginal_mu = marginal_mu[..., 0]
    marginal_var = marginal_var[..., 0]

    marginal_mu = np.transpose(marginal_mu, [1, 0, 2])
    marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])

    out_block_dim = get_block_dim(out_block_arr[0])

    chex.assert_shape(marginal_mu, [N, num_latents,  out_block_dim])
    chex.assert_shape(marginal_var, [N, num_latents, out_block_dim, out_block_dim])

    return marginal_mu, marginal_var


# Linear Transform Entry Point
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):
    """ Recursively compute the transformed linear marginal. """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    return linear_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block, whiten, XS=None
    )

    
# ===============================================================================================
# ===============================================================================================
# ========================================  ENTRY POINTs ========================================
# ===============================================================================================
# ===============================================================================================


# list of linear priors
@dispatch(ApproximatePosterior, Likelihood, list, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, list, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):
    """ 
    Recursively compute the transformed linear marginal.  

    Prior is a list of transforms to be comptued Recursively. Simply loop through and collect the results.

    Note we do not use batching and this method will only really be used when the transforms in the list are 
        different, and hence batching wont be applicable anyway.

    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    mu_list, var_list = [], []
    for p in prior:
        mu_p, var_p  = evoke('marginal_blocks', approximate_posterior, likelihood, p, whiten=whiten)(
            data, q_m, q_S_chol, approximate_posterior, likelihood, p, out_block, whiten
        ) 

        mu_list.append(mu_p)
        var_list.append(var_p)

    return mu_list, var_list

# ============================ SINGLE OUTPUT APPROXIMATE POSTERIOR ENTRY POINT ============================

@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=True)
@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    tbd.
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    out_block_type = likelihood.block_type

    sparsity = prior.sparsity

    # we are only processing the linear part so we can assume that it is linear
    mu, var = evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_type, whiten
    ) 

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

# ============================ MEANFIELD APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    .
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # find out if the model is linear or not
    model_type = get_model_type(prior)


    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # get output block type. NonLinear transforms are applied elementwise so only need the likelihood
    #   block size
    out_block: Block = likelihood.block_type

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block, whiten
    ) 

    return val[0], val[1]

# ============================ FULL GAUSSIAN APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # When using FullGaussian approximate posteriors we require the block covariances
    #  to compute the resulting ELLs

    # TODO: this is hack
    if likelihood.block_type == Block.DIAGONAL:
        out_block_type = Block.LATENT
    else:
        out_block_type = likelihood.block_type

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block_type, whiten
    ) 

    return val[0], val[1]



