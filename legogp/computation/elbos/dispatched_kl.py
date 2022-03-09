import chex
import jax
import jax.numpy as np

from ...dispatch import dispatch, evoke
from .kullback_leiblers import gaussian_cholesky_kl
from ...transforms import Transform, Independent
from ...utils.batch_utils import batch_over_module_types
from ..matrix_ops import cholesky, add_jitter
from ...settings import jitter
from .prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ, prior_mean_Z, prior_mean_X



@dispatch('GaussianApproximatePosterior', 'GPPrior')
def kullback_leibler(approximate_posterior, prior):

    Z = prior.sparsity.Z
    # We need to index prior.(mean/covar) because they return a 3d array for consistency to multi-output priors
    covar_2 = prior.covar(Z, Z)[0]
    covar_chol_2 = cholesky(add_jitter(covar_2, jitter))

    return gaussian_cholesky_kl(
        approximate_posterior.m,
        approximate_posterior.S_chol,
        prior.mean(Z)[0],
        covar_chol_2
    )

@dispatch('MeanFieldApproximatePosterior', Independent)
def kullback_leibler(approximate_posterior, prior):

    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors

    kl_arr = batch_over_module_types(
        evoke_name = 'kullback_leibler',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, latents_arr],
        fn_params = [approx_posteriors_arr, latents_arr],
        fn_axes = [0, 0],
        dim = len(latents_arr),
        out_dim  = 1 
    )

    chex.assert_shape(kl_arr, [len(latents_arr)])

    return np.sum(kl_arr)


@dispatch('MeanFieldApproximatePosterior', Transform)
def kullback_leibler(approximate_posterior, prior):

    latents = prior.latent_obj

    return evoke('kullback_leibler', approximate_posterior, latents)(
        approximate_posterior, latents
    )

@dispatch('FullGaussianApproximatePosterior', Transform)
def kullback_leibler(approximate_posterior, prior):
    # Reshape mean_2 to have same shape as approxiamte posterior
    mean_2_blocks = prior_mean_Z(prior)
    covar_2_blocks = prior_covar_ZZ(prior)

    mean_2 = np.vstack(mean_2_blocks)
    covar_2 = jax.scipy.linalg.block_diag(*covar_2_blocks)
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
