from .expected_log_likelihoods import *
from .. import settings
from .. batching import Batched
from .. import settings

def linear_transform_ell(X, Y, likelihood: 'Likelihood', transform, approx_posteriors):
    """
    Args:
        X:
        Y:
        likelihood:
        approx_posteriors: List of GaussianApproximatePosteriors
    """

    #Collect marginals of all approximate posteriors

    weights = transform.W_p
    latents = transform.latents_p
    num_latents = len(latents)

    if False:
        latent_mean_arr = approx_posteriors.precomputed_marginal_mean_arr
        latent_var_arr = approx_posteriors.precomputed_marginal_var_arr
    else:
        latent_mean_arr = approx_posteriors.approx_posteriors[0].m[None, :]
        latent_var_arr = approx_posteriors.approx_posteriors[0].S_diag[None, :]

    weights =  weights[None, :]

    if True:
        mean_p = weights @ latent_mean_arr[..., 0]
        var_p = (weights**2) @ latent_var_arr[..., 0]

        mean_p  = mean_p[0][..., None]
        var_p  = var_p[0][ ..., None]
    else:
        mean_p = latent_mean_arr[0]
        var_p = latent_var_arr[0]

    #chex.assert_equal(mean_p.shape, latent_mean_arr.shape)
    #chex.assert_equal(var_p.shape, latent_var_arr.shape)

    #mean_p = np.sum(mean_p, axis=0)
    #var_p = np.sum(var_p, axis=0)

    chex.assert_equal(mean_p.shape, Y.shape)
    chex.assert_equal(var_p.shape, Y.shape)


    return gaussian_expected_log_likelihood(X, Y, likelihood.variance, mean_p, var_p)


def non_linear_transform_ell(X, Y, likelihood: 'Likelihood', transform, approx_posteriors):
    latent_mean_arr = approx_posteriors.precomputed_marginal_mean_arr
    latent_var_arr = approx_posteriors.precomputed_marginal_var_arr

    sample_arr = approx_posteriors.sample_from_precomputed(settings.monte_carlo_training_samples)


    def vmap_over_samples(Y, mean_arr, likelihood, transform):
        T_f = transform.forward(mean_arr)

        chex.assert_equal(Y.shape, T_f.shape)


        return likelihood.batched_log_likelihood(
            Y,
            T_f
            
        )

    total_ell = jax.vmap(vmap_over_samples, (None, 0, None, None), 0)(Y, sample_arr, likelihood, transform)

    ell = np.mean(total_ell)


    return ell

