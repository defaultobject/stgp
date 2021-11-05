import jax
import jax.numpy as np
import chex
from .. import settings

def linear_predictor(XS, X, likelihood, transform, approx_posterior):
    weights = transform.W_p
    latents = transform.latents_p
    num_latents = len(latents)

    if False:
        latent_mean_arr = approx_posterior.precomputed_marginal_mean_arr
        latent_var_arr = approx_posterior.precomputed_marginal_var_arr
    else:
        latent_mean_arr = approx_posterior.approx_posteriors[0].m[None, :]
        latent_var_arr = approx_posterior.approx_posteriors[0].S_diag[None, :]

    weights =  weights[:, None, None]


    print('weights: ', weights)

    if True:
        mean_p = weights @ latent_mean_arr[..., 0]
        var_p = (weights**2) @ latent_var_arr[..., 0]
        mean_p  = mean_p[0][..., None]
        var_p  = var_p[0, ..., None]
    else:
        mean_p = latent_mean_arr * weights
        var_p = latent_var_arr * (weights **2)

    chex.assert_equal(mean_p.shape, latent_mean_arr.shape)
    chex.assert_equal(var_p.shape, latent_var_arr.shape)

    mean_p = np.sum(mean_p, axis=0)
    var_p = np.sum(var_p, axis=0) + likelihood.variance

    print('var_p: ', latent_var_arr)
    print('var: ', likelihood.variance)


    return mean_p, var_p

def non_linear_predictor(XS, X, likelihood, transform, approx_posterior):
    latent_mean_arr = approx_posterior.precomputed_marginal_mean_arr
    latent_var_arr = approx_posterior.precomputed_marginal_var_arr

    sample_arr = approx_posterior.sample_from_precomputed(settings.monte_carlo_prediction_samples)


    def vmap_over_samples(mean_arr, likelihood, transform):
        T_f = transform.forward(mean_arr)

        mu = likelihood.conditional_mean(T_f)

        return mu, mu


    mean, var = jax.vmap(vmap_over_samples, (0, None, None), (0, 0))(sample_arr, likelihood, transform)

    mean = np.mean(mean, axis=0)
    var = np.ones_like(mean)

    return mean, var
