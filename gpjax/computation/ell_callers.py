from .expected_log_likelihoods import *
from .. import settings
from .. batching import Batched

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

    latent_mean_arr = approx_posteriors.precomputed_marginal_mean_arr
    latent_var_arr = approx_posteriors.precomputed_marginal_var_arr

    weights =  weights[:, None, None]

    mean_p = latent_mean_arr * weights
    var_p = latent_var_arr * (weights **2)

    chex.assert_equal(mean_p.shape, latent_mean_arr.shape)
    chex.assert_equal(var_p.shape, latent_var_arr.shape)

    mean_p = np.sum(mean_p, axis=0)
    var_p = np.sum(var_p, axis=0)

    chex.assert_equal(mean_p.shape, (X.shape[0], 1))
    chex.assert_equal(var_p.shape, (X.shape[0], 1))

    return gaussian_expected_log_likelihood(X, Y, likelihood.variance, mean_p, var_p)
