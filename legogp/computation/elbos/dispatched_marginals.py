import chex
import jax.numpy as np

from ...dispatch import dispatch, evoke
from ...transforms import LinearTransform, Independent
from ...utils.batch_utils import batch_over_module_types
from ..marginals import gaussian_conditional_diagional
from .prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ
from ..matrix_ops import diagonal_from_cholesky

@dispatch('GaussianApproximatePosterior', 'GPPrior', 'NoSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    return approximate_posterior.m, diagonal_from_cholesky(approximate_posterior.S_chol)

@dispatch('prediction', 'GaussianApproximatePosterior', 'GPPrior', 'NoSparsity')
def marginal(XS, X, approximate_posterior, prior, sparsity):
    # Compute Kzz, Kxz, Kxs_diag, mean_x, mean_xs
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


@dispatch('MeanFieldApproximatePosterior', Independent)
def marginal(X, approximate_posterior, prior):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()

    num_latents = len(sparsity_arr)
    N = X.shape[0]

    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_params = [X, approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_axes = [None, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch('prediction', 'MeanFieldApproximatePosterior', Independent)
def marginal(XS, X, approximate_posterior, prior):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()

    num_latents = len(sparsity_arr)
    N = XS.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = ['prediction'],
        module_arr = [approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_params = [XS, X, approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch('MeanFieldApproximatePosterior', LinearTransform)
def marginal(X, approximate_posterior, prior):

    latents = prior.latents

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

@dispatch('prediction', 'MeanFieldApproximatePosterior', LinearTransform)
def marginal(XS, X, approximate_posterior, prior):

    latents = prior.latents

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
