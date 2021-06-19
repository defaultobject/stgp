import jax.numpy as np
import chex

def linear_predictor(XS, X, likelihood, transform, approx_posterior):
    weights = transform.W_p
    latents = transform.latents_p
    num_latents = len(latents)

    latent_mean_arr = approx_posterior.precomputed_marginal_mean_arr
    latent_var_arr = approx_posterior.precomputed_marginal_var_arr

    weights =  weights[:, None, None]

    mean_p = latent_mean_arr * weights
    var_p = latent_var_arr * (weights **2)

    chex.assert_equal(mean_p.shape, latent_mean_arr.shape)
    chex.assert_equal(var_p.shape, latent_var_arr.shape)

    mean_p = np.sum(mean_p, axis=0)
    var_p = np.sum(var_p, axis=0) + likelihood.variance

    return mean_p, var_p
