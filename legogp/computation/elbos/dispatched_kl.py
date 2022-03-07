import chex
import jax
import jax.numpy as np

from ...dispatch import dispatch, evoke
from .kullback_leiblers import gaussian_cholesky_kl
from ...transforms import Transform, Independent
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


@dispatch('MeanFieldApproximatePosterior', Transform)
def kullback_leibler(X, approximate_posterior, prior):

    latents = prior.latent_obj

    return evoke('kullback_leibler', approximate_posterior, latents)(
        X, approximate_posterior, latents
    )

@dispatch('FullGaussianApproximatePosterior', Transform)
def kullback_leibler(X, approximate_posterior, prior):
    # Get latents
    latent_obj = prior.latent_obj

    mean_2 = latent_obj.mean(X)

    # Reshape mean_2 to have same shape as approxiamte posterior
    Q = mean_2.shape[0]
    N = mean_2.shape[1]
    mean_2 = np.reshape(mean_2, [Q * N , 1])

    covar_2 = jax.scipy.linalg.block_diag(*latent_obj.covar(X, X))
    covar_chol_2 = cholesky(add_jitter(covar_2, jitter))

    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    chex.assert_equal(m.shape, mean_2.shape)
    chex.assert_equal(S_chol.shape, covar_chol_2.shape)

    return gaussian_cholesky_kl(
        approximate_posterior.m,
        approximate_posterior.S_chol,
        mean_2,
        covar_chol_2
    )
