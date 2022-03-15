import chex
import jax
import jax.numpy as np

from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform
from ....approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity

@dispatch('GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', 'NoSparsity')
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    return approximate_posterior.m, diagonal_from_cholesky(approximate_posterior.S_chol)

@dispatch('GaussianApproximatePosterior', BlockDiagonalLikelihood, 'GPPrior', 'NoSparsity')
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    block_size = likelihood.block_size
    return  block_from_vec(approximate_posterior.m, block_size), block_diagonal_from_cholesky(approximate_posterior.S_chol, block_size)

@dispatch('GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', 'FullSparsity')
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    # TODO: this is a hack
    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity)

    return fn(
        X, X, approximate_posterior, likelihood, prior, sparsity
    )

@dispatch('prediction', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', FreeSparsity)
def marginal(XS, X, approximate_posterior, likelihood, prior, sparsity):
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    return gaussian_conditional_diagional(
        XS, 
        X, 
        prior.covar(sparsity.Z, sparsity.Z)[0], 
        prior.covar(XS, sparsity.Z)[0], 
        prior.var(XS)[0], 
        m,
        S_chol,
        prior.mean(sparsity.Z)[0],
        prior.mean(XS)[0],
    )

@dispatch('full_prediction', 'GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', FreeSparsity)
def marginal(XS, X, approximate_posterior, likelihood, prior, sparsity):
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    return gaussian_conditional(
        XS, 
        X, 
        prior.covar(sparsity.Z, sparsity.Z)[0], 
        prior.covar(XS, sparsity.Z)[0], 
        prior.covar(XS, XS)[0], 
        m,
        S_chol,
        prior.mean(sparsity.Z)[0],
        prior.mean(XS)[0],
    )


@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent)
def marginal(X, approximate_posterior, likelihood, prior):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = X.shape[0]

    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.
    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [X, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    #chex.assert_shape(marginal_mu, [num_latents, N, 1])
    #chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch('prediction', MeanFieldApproximatePosterior, ProductLikelihood, Independent)
def marginal(XS, X, approximate_posterior, likelihood, prior, inference, diagonal):
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

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [evoke_params],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [XS, X, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    #chex.assert_shape(marginal_mu, [num_latents, N, 1])
    #chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch(MeanFieldApproximatePosterior, Likelihood, LinearTransform)
def marginal(X, approximate_posterior, likelihood, prior):

    latents = prior.latent_obj

    num_latents = len(latents.latents)
    N = X.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', approximate_posterior, latents)(
        X, approximate_posterior, latents
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch(MeanFieldApproximatePosterior, Likelihood, NonLinearTransform)
def marginal(X, approximate_posterior, prior):
    """ 
    Computation of the ELL with a non-linear transform is computed using monte-carlo. Therefore we return the (untransformed)
    latents here so they can be used to perform the monte-carlo approximation.
    """
    latents = prior.latent_obj

    return   evoke('marginal', approximate_posterior, latents)(
        X, approximate_posterior, latents
    ) 

@dispatch('prediction', MeanFieldApproximatePosterior, LinearTransform)
def marginal(XS, X, approximate_posterior, prior, inference):

    latents = prior.latent_obj

    num_latents = len(latents.latents)
    N = XS.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', 'prediction', approximate_posterior, latents)(
        XS, X, approximate_posterior, latents
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch('prediction', MeanFieldApproximatePosterior, NonLinearTransform)
def marginal(XS, X, approximate_posterior, prior, inference):
    latents = prior.latent_obj

    latent_mu, latent_var =   evoke('marginal', 'prediction', approximate_posterior, latents)(
        XS, X, approximate_posterior, latents
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
    
    second_moment =  mu**2

    mu = np.mean(mu, axis=0)
    second_moment = np.mean(second_moment, axis=0)

    mu = np.transpose(mu, [1, 0, 2])
    second_moment = np.transpose(second_moment, [1, 0, 2])


    var = second_moment - np.square(mu)
    
    return mu, var
