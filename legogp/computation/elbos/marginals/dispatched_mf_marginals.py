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
"""
import chex
import jax
import jax.numpy as np

from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo


# ================================== Dispatched q(f) ==============================

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity')
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity):
    """ Catch all for single latent functions """
    return q_m, q_S

@dispatch(ApproximatePosterior, DiagonalLikelihood, 'GPPrior', 'SpatialSparsity')
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity):
    """ Compute diagonal spatial sparsity conditional """

    data_Z = sparsity.raw_Z
    mu, var = evoke('spatial_conditional', data, data_Z, 'BASE_SDE_GP', prior)(
        data,
        data_Z,
        q_m,
        q_S,
        prior,
        True
    )

    mu = np.reshape(mu, [-1, 1])
    var = np.reshape(var, [-1, 1])

    return mu, var

@dispatch(ApproximatePosterior, BlockDiagonalLikelihood, 'GPPrior', 'SpatialSparsity')
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity):
    """ Compute block diagonal spatial sparsity conditional """

    # Ensure rank 2
    q_m = np.reshape(q_m, [data.Nt, -1])

    data_Z = sparsity.raw_Z
    mu, var = evoke('spatial_conditional', data, data_Z, 'BASE_SDE_GP', prior)(
        data,
        data_Z,
        q_m,
        q_S,
        prior,
        False
    )

    return mu, var

@dispatch('GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', 'FullSparsity')
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity):
    """ The marginal q(f) here is the same as the predictive distribution q(f*).  """

    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity)

    return fn(
        data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity
    )

@dispatch('prediction', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', Sparsity)
def marginal(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity):
    """ Computes the diagonal of q(f) = ∫ p(f | u) q(u) du """
    chex.assert_rank([m, S_chol], [2, 2])
    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

    return gaussian_conditional_diagional(
        XS, 
        data.X, 
        prior.covar(sparsity.Z, sparsity.Z)[0], 
        prior.covar(XS, sparsity.Z)[0], 
        prior.var(XS)[0], 
        m,
        S_chol,
        prior.mean(sparsity.Z)[0],
        prior.mean(XS)[0],
    )

@dispatch('full_prediction', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', FreeSparsity)
def marginal(XS, data, approximate_posterior, likelihood, prior, sparsity):
    """ Computes the full q(f) = ∫ p(f | u) q(u) du """

    raise NotImplementedError()
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    return gaussian_conditional(
        XS, 
        data.X, 
        prior.covar(sparsity.Z, sparsity.Z)[0], 
        prior.covar(XS, sparsity.Z)[0], 
        prior.covar(XS, XS)[0], 
        m,
        S_chol,
        prior.mean(sparsity.Z)[0],
        prior.mean(XS)[0],
    )

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior')
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior):
    sparsity = prior.sparsity
    mu, var = evoke('marginal', approximate_posterior, likelihood, prior, sparsity)(
        data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity
    ) 

    return mu, var


@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = data.X.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.
    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [data, q_m, q_S, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    #chex.assert_shape(marginal_mu, [num_latents, N, 1])
    #chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, Independent)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    # When prediction we can just directly get the raw params
    q_m, q_S = approximate_posterior.get_variational_params()

    num_latents = len(sparsity_arr)
    N = XS.shape[0]

    if diagonal:
        evoke_params = 'prediction'
    else:
        evoke_params = 'full_prediction'

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [evoke_params],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [XS, data, q_m, q_S, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    #chex.assert_shape(marginal_mu, [num_latents, N, 1])
    #chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch('prediction', MeanFieldConjugateGaussian, ProductLikelihood, Independent)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = XS.shape[0]

    if diagonal:
        evoke_params = 'prediction'
    else:
        evoke_params = 'full_prediction'

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [evoke_params],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [XS, data, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    #chex.assert_shape(marginal_mu, [num_latents, N, 1])
    #chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch(MeanFieldApproximatePosterior, Likelihood, LinearTransform)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior):

    latents = prior.latent_obj

    num_latents = len(latents.latents)
    N = data.X.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', approximate_posterior, likelihood, latents)(
        data, q_m , q_S, approximate_posterior, likelihood, latents
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch(MeanFieldApproximatePosterior, Likelihood, NonLinearTransform)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior):
    """ 
    Computation of the ELL with a non-linear transform is computed using monte-carlo. Therefore we return the (untransformed)
    latents here so they can be used to perform the monte-carlo approximation.
    """
    latents = prior.latent_obj

    return   evoke('marginal', approximate_posterior, likelihood, latents)(
        data, q_m, q_S, approximate_posterior, likelihood, latents
    ) 

# ========================= Predictions =========================


@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, LinearTransform)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal):
    if diagonal is False:
        raise NotImplementedError()

    latents = prior.latent_obj

    num_latents = len(latents.latents)
    N = XS.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', 'prediction', approximate_posterior, likelihood, latents)(
        XS, data, approximate_posterior, likelihood, latents, inference, diagonal
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, NonLinearTransform)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal):
    if diagonal is False:
        raise NotImplementedError()

    latents = prior.latent_obj

    latent_mu, latent_var =  evoke('marginal', 'prediction', approximate_posterior, likelihood, latents)(
        XS, data, approximate_posterior, likelihood, latents, inference, diagonal
    ) 

    vmaped_prior_forard =  jax.vmap(prior.forward, [1], 0)

    mu = mv_indepentdent_monte_carlo(
        lambda f, fn: fn(f),
        latent_mu,
        latent_var,
        fn_args=[vmaped_prior_forard],
        generator = inference.generator, 
        num_samples = inference.prediction_samples,
        average=False
    )

    # Ensure correct shape
    mu = np.reshape(mu, [inference.prediction_samples, XS.shape[0], prior.output_dim, 1])
    
    second_moment =  mu**2

    mu = np.mean(mu, axis=0)
    second_moment = np.mean(second_moment, axis=0)

    mu = np.transpose(mu, [1, 0, 2])
    second_moment = np.transpose(second_moment, [1, 0, 2])


    var = second_moment - np.square(mu)
    
    return mu, var
