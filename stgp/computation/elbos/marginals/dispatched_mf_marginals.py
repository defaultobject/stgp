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
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part


# ================================== Dispatched q(f) ==============================

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, whiten):
    """ Catch all for single latent functions """
    return q_m, q_S

@dispatch(ApproximatePosterior, DiagonalLikelihood, 'GPPrior', 'NoSparsity', whiten=True)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, whiten):
    chex.assert_rank(q_S_chol, 2)
    chex.assert_shape(q_S_chol, [q_m.shape[0], q_m.shape[0]])

    Kzz = prior.covar(sparsity.Z, sparsity.Z)
    Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    return Kzz_chol @ q_m, diagonal_from_cholesky(Kzz_chol @ q_S_chol)

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

@dispatch('prediction', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', Sparsity, whiten=False)
def marginal(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, whiten):
    """ Computes the diagonal of q(f) = ∫ p(f | u) q(u) du """
    chex.assert_rank([m, S_chol], [2, 2])
    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

    return gaussian_conditional_diagional(
        XS, 
        data.X, 
        prior.covar(sparsity.Z, sparsity.Z), 
        prior.covar(XS, sparsity.Z), 
        prior.var(XS), 
        m,
        S_chol,
        prior.mean(sparsity.Z),
        prior.mean(XS),
    )

@dispatch('prediction', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', Sparsity, whiten=True)
def marginal(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, whiten):
    """ Computes the diagonal of q(f) = ∫ p(f | u) q(u) du """
    chex.assert_rank([m, S_chol], [2, 2])
    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

    #TODO: only works with zero mean gps
    return whitened_gaussian_conditional_diagional(
        XS, 
        data.X, 
        prior.covar(sparsity.Z, sparsity.Z), 
        prior.covar(XS, sparsity.Z), 
        prior.var(XS)[:, 0], 
        m,
        S_chol
    )

@dispatch('full_prediction', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', FreeSparsity)
def marginal(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity):
    """ Computes the full q(f) = ∫ p(f | u) q(u) du """

    chex.assert_rank([m, S_chol], [2, 2])
    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

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


@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=True)
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=False)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, whiten):
    latents_arr = prior.parent
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = data.X.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]
    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.
    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_params = [data, q_m, q_S, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_axes = [None, 0, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    #chex.assert_shape(marginal_mu, [num_latents, N, 1])
    #chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var



@dispatch(MeanFieldApproximatePosterior, Likelihood, LinearModel, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, LinearModel, whiten=False)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, whiten):
    breakpoint()

    base_prior = prior.base_prior

    output_dim = prior.output_dim
    N = data.X.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', approximate_posterior, likelihood, base_prior, whiten=whiten)(
        data, q_m , q_S, approximate_posterior, likelihood, base_prior, whiten
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [output_dim, N, 1])
    chex.assert_shape(marginal_var, [output_dim, N, 1])

    return marginal_mu, marginal_var


@dispatch(MeanFieldApproximatePosterior, Likelihood, NonLinearModel, whiten=False)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, whiten):
    """ 
    Computation of the ELL with a non-linear transform is computed using monte-carlo. Therefore we return the (untransformed)
    latents here so they can be used to perform the monte-carlo approximation.
    """
    latents = prior.parent

    return   evoke('marginal', approximate_posterior, likelihood, latents, whiten=whiten)(
        data, q_m, q_S, approximate_posterior, likelihood, latents, whiten
    ) 


@dispatch(MeanFieldApproximatePosterior, Likelihood, Aggregate)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior):

    # we need to compute the full predictive distributions for each aggregated group

    latents_arr = prior.latent_obj.latents
    likelihood_arr = likelihood.likelihood_arr
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()

    num_latents = prior.num_latents

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    
    # Compute q(f) for each site
    site_fn = lambda X: batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = ['full_prediction'],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [X, data, q_m, q_S, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, 0, 0, 0, 0, 0, 0],
        dim = num_latents,
        out_dim  = 2
    )

    marginal_mu, marginal_var = jax.vmap(site_fn, (0, ))(data.X)

    # fix shapes
    marginal_mu = marginal_mu[..., 0]

    group_size = marginal_mu.shape[2]

    marginal_mu = np.sum(marginal_mu, axis=2)/group_size
    marginal_var = np.sum(np.sum(marginal_var, axis=2), axis=2)/(group_size*group_size)

    return (marginal_mu.T)[..., None], (marginal_var.T)[..., None]


@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, whiten):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal', approximate_posterior, likelihood, LinearModel, whiten=whiten)(
        data, q_m, q_S, approximate_posterior, likelihood, linear_model_part, whiten
    ) 
    breakpoint()

# ========================= Predictions =========================

@dispatch('latents', MeanFieldApproximatePosterior, Likelihood, Transform)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal):
    latents = prior.latent_obj

    mu, var = evoke('marginal', 'prediction', approximate_posterior, likelihood, latents)(
        XS, data, approximate_posterior, likelihood, latents, inference, diagonal 
    )

    return mu.T, var.T

@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=False)
@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=True)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    # When predicting we can just directly get the raw params
    q_m, q_S_chol = approximate_posterior.get_variational_params()

    num_latents = len(sparsity_arr)
    N = XS.shape[0]

    if diagonal:
        evoke_params = 'prediction'
    else:
        evoke_params = 'full_prediction'

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]
    whiten_arr = [whiten for q in range(num_latents)]

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [evoke_params],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_params = [XS, data, q_m, q_S_chol, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_axes = [None, None, 0, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    #chex.assert_shape(marginal_mu, [num_latents, N, 1])
    #chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch('prediction', MeanFieldConjugateGaussian, ProductLikelihood, Independent, whiten=False)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):
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

@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, LinearModel, whiten=False)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):
    if diagonal is False:
        raise NotImplementedError()

    base_prior = prior.base_prior

    num_latents = base_prior.output_dim
    N = XS.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', 'prediction', approximate_posterior, likelihood, base_prior, whiten)(
        XS, data, approximate_posterior, likelihood, base_prior, inference, whiten, diagonal
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, Aggregate, whiten=False)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):
    if diagonal is False:
        raise NotImplementedError()

    # When predicting we can just directly get the raw params
    q_m, q_S_chol = approximate_posterior.get_variational_params()
    # we need to compute the full predictive distributions for each aggregated group

    latents_arr = prior.latent_obj.latents
    likelihood_arr = likelihood.likelihood_arr
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()

    num_latents = prior.num_latents

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    
    # Compute q(f) for each site
    site_fn = lambda X: batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = ['full_prediction'],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [X, data, q_m, q_S_chol, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, 0, 0, 0, 0, 0, 0],
        dim = num_latents,
        out_dim  = 2
    )

    marginal_mu, marginal_var = jax.vmap(site_fn, (0, ))(XS)

    # fix shapes
    marginal_mu = marginal_mu[..., 0]

    group_size = marginal_mu.shape[2]

    marginal_mu = np.sum(marginal_mu, axis=2)/group_size
    marginal_var = np.sum(np.sum(marginal_var, axis=2), axis=2)/(group_size*group_size)

    return (marginal_mu.T)[..., None], (marginal_var.T)[..., None]

@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, Transform, whiten=False)
@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, Transform, whiten=True)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    return   evoke('marginal', 'prediction', approximate_posterior, likelihood, model_type, whiten)(
        XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal
    ) 

@dispatch('samples', MeanFieldApproximatePosterior, ProductLikelihood, NonLinearModel, whiten=False)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):

    if diagonal is False:
        raise NotImplementedError()

    latents = prior.base_prior
    model_type = get_model_type(prior)

    # compute predictions of the (Gaussian) latent functions
    latent_mu, latent_var =  evoke('marginal', 'prediction', approximate_posterior, likelihood, latents, whiten)(
        XS, data, approximate_posterior, likelihood, latents, inference, whiten, diagonal
    ) 

    vmaped_prior_forard =  jax.vmap(prior.forward, [1], 0)

    # sample and push through the nonlinear transform
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

    return mu

@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, NonLinearModel, whiten=False)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):
    base_prior = prior.base_prior

    model_type = get_model_type(prior)

    #compute samples
    mu =  evoke('marginal', 'samples', approximate_posterior, likelihood, model_type, whiten)(
        XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal
    ) 
    
    second_moment =  mu**2

    mu = np.mean(mu, axis=0)
    second_moment = np.mean(second_moment, axis=0)

    mu = np.transpose(mu, [1, 0, 2])
    second_moment = np.transpose(second_moment, [1, 0, 2])


    var = second_moment - np.square(mu)
    
    return mu, var

# =================== COVAR PREDICTION ===================

@dispatch('prediction_covar', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', Sparsity)
def marginal(X1, X2, data, approximate_posterior, likelihood, prior, sparsity):
    m, S_chol  = approximate_posterior.m, approximate_posterior.S_chol
    K12 = prior.kernel.K(X1, X2)
    K1z = prior.kernel.K(X1, sparsity.Z)
    Kz2 = prior.kernel.K(sparsity.Z, X2)
    Kzz = prior.kernel.K(sparsity.Z, sparsity.Z)

    return gaussian_conditional_covar(
        X1, X2, sparsity.Z,
        Kzz, 
        K1z,
        Kz2,
        K12,
        m,
        S_chol
    )

@dispatch('prediction_covar', MeanFieldApproximatePosterior, ProductLikelihood, Transform)
def marginal(X1, X2, data, approximate_posterior, likelihood, prior, inference):
    latents = prior.latent_obj

    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)


    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    # Compute q(f) for each output
    marginal_covar = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = ['prediction_covar'],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [X1, X2, data, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, None, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 1
    )

    return marginal_covar

