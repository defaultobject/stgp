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

    with Batched(latents) as latents, Batched(approx_posteriors) as approx_posteriors:
        def batched_marginals(X, w_q, latent, latent_vars, approx_posterior, approx_posterior_vars):
            latent.set_vars(latent_vars)
            approx_posterior.set_vars(approx_posterior_vars)

            latent = latent.get_obj()
            approx_posterior = approx_posterior.get_obj()


            #todo cache marginals?
            mean_q, var_q = approx_posterior.marginal(
                X,
                latent.kernel[0],
                latent.sparsity
            )

            return mean_q * w_q, var_q * (w_q ** 2)


        mean_p, var_p = jax.vmap(batched_marginals, (None, 0, None, 0, None, 0), (0, 0))(X, weights, latents, latents.get_vars(), approx_posteriors, approx_posteriors.get_vars())

        chex.assert_equal(mean_p.shape, (num_latents, X.shape[0], 1))
        chex.assert_equal(var_p.shape, (num_latents, X.shape[0], 1))

    mean_p = np.sum(mean_p, axis=0)
    var_p = np.sum(var_p, axis=0)

    chex.assert_equal(mean_p.shape, (X.shape[0], 1))
    chex.assert_equal(var_p.shape, (X.shape[0], 1))

    return gaussian_expected_log_likelihood(X, Y, likelihood.variance, mean_p, var_p)
