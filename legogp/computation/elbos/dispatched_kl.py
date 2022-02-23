import chex
import jax.numpy as np

from ...dispatch import dispatch, evoke
from .kullback_leiblers import gaussian_cholesky_kl
from ...transforms import LinearTransform, Independent
from ...utils.batch_utils import batch_over_module_types
from ..matrix_ops import cholesky, add_jitter
from ...settings import jitter



@dispatch('GaussianApproximatePosterior', 'GPPrior')
def kullback_leibler(X, approximate_posterior, prior):

    # We need to index prior.(mean/covar) because they return a 3d array for consistency to multi-output priors
    covar_2 = prior.covar(X, X)[0]
    covar_chol_2 = cholesky(add_jitter(covar_2, jitter))

    return gaussian_cholesky_kl(
        approximate_posterior.m,
        approximate_posterior.S_chol,
        prior.mean(X)[0],
        covar_chol_2
    )

@dispatch('MeanFieldApproximatePosterior', Independent)
def kullback_leibler(X, approximate_posterior, prior):

    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors

    kl_arr = batch_over_module_types(
        evoke_name = 'kullback_leibler',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, latents_arr],
        fn_params = [X, approx_posteriors_arr, latents_arr],
        fn_axes = [None, 0, 0],
        dim = len(latents_arr),
        out_dim  = 1 
    )

    chex.assert_shape(kl_arr, [len(latents_arr)])

    return np.sum(kl_arr)


@dispatch('MeanFieldApproximatePosterior', LinearTransform)
def kullback_leibler(X, approximate_posterior, prior):

    latents = prior.latents

    return evoke('kullback_leibler', approximate_posterior, latents)(
        X, approximate_posterior, latents
    )
