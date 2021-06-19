"""Variational inference class."""
from . import Inference
from .. import settings
from .. batching import Batched

import jax
import jax.numpy as np
import chex

class Variational(Inference):
    """Variational inference class."""

    def predict(self, XS, X, likelihood, kernel, sparsity, approximate_posterior, diagonal):
        num_latents = len(kernel)
        if True or settings.use_loop_mode:
            mean_arr, var_arr = [], []

            for q in range(num_latents):
                mean_q, var_q = approximate_posterior[q].predict(
                    XS,
                    X,
                    likelihood[q],
                    kernel[q],
                    sparsity[q],
                    diagonal
                )


                mean_arr.append(mean_q)
                var_arr.append(var_q)

            return np.array(mean_arr), np.array(var_arr)



    def ELBO(self, X, Y, likelihood, prior, approximate_posterior, minibatch):
        """
        Args:
            X: NxD input
            Y: NXP outputs
            likelihood: Array of P likelihoods
            prior: 
            approximate_posterior: Q approximate posteriors
        """

        #num_latents = len(kernel)
        num_outputs = Y.shape[1]

        approximate_posterior.precompute_marginals(X, prior)

        #get the p transforms, one for each output
        transforms = prior.get_batches()

        if settings.use_loop_mode:

            ell = 0.0
            for p in range(num_outputs):
                _ell_p = ell_p_callable(
                    X, 
                    Y[:, p][:, None], 
                    likelihood[p],
                    transforms[p],
                    approximate_posterior,
                )

                ell += _ell_p
        else:

            with Batched(likelihood) as likelihood, Batched(transforms) as transforms:

                def batched_elbo(X, Y, lik, lik_vars, transform, transform_vars, approximate_posterior):

                    N = X.shape[0]
                    Y = Y[:, None]

                    lik.set_vars(lik_vars)
                    transform.set_vars(transform_vars)

                    elbo_q = ell_p_callable(
                        X, 
                        Y, 
                        lik.get_obj(),
                        transform.get_obj(),
                        approximate_posterior
                    )

                    return elbo_q

                ell = jax.vmap(batched_elbo, (None, 1, None, 0, None, 0,  None), 0)(X, Y, likelihood, likelihood.get_vars(), transforms, transforms.get_vars(), approximate_posterior)

                ell = np.sum(ell) 


        KL = approximate_posterior.KL(X, prior)

        elbo = ell - KL

        chex.assert_rank(elbo, 0)
        return elbo


def ell_p_callable(X, Y, likelihood_p, transform_p, approximate_posterior):
    """ Compute the expected log likelihood for output p """

    ell_q = approximate_posterior.ELL(
        X, 
        Y, 
        likelihood_p,
        transform_p
    )

    #assert scalar
    chex.assert_rank(ell_q, 0)

    return ell_q

