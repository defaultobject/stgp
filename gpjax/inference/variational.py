"""Variational inference class."""
from . import Inference
from .. import settings
from .. batching import Batched

import jax
import jax.numpy as np


class Variational(Inference):
    """Variational inference class."""

    def predict(self, XS, X, likelihood, kernel, sparsity, approximate_posterior):
        num_latents = len(kernel)
        if True or settings.use_loop_mode:
            mean_arr, var_arr = [], []

            for q in range(num_latents):
                mean_q, var_q = approximate_posterior[q].predict(
                    XS,
                    X,
                    likelihood[q],
                    kernel[q],
                    sparsity[q]
                )

                mean_arr.append(mean_q)
                var_arr.append(var_q)

            return np.array(mean_arr), np.array(var_arr)



    def ELBO(self, X, Y, likelihood, kernel, sparsity, approximate_posterior):

        def _elbo_q(X, Y, likelihood, kernel, sparsity, approximate_posterior):
            ell_q = approximate_posterior.ELL(
                X, 
                Y, 
                likelihood,
                kernel,
                sparsity
            )

            kl_q = approximate_posterior.KL(
                X, 
                kernel,
                sparsity
            )
            elbo_q = ell_q - kl_q

            return elbo_q



        num_latents = len(kernel)
        if settings.use_loop_mode:
            elbo = 0.0

            for q in range(num_latents):
                elbo_q = _elbo_q(
                    X, 
                    Y[:, q][:, None], 
                    likelihood[q],
                    kernel[q],
                    sparsity[q],
                    approximate_posterior[q]
                )

                elbo += elbo_q
        else:

            with Batched(likelihood) as likelihood, \
                Batched(kernel) as kernel, \
                Batched(approximate_posterior) as approximate_posterior, \
                Batched(sparsity) as sparsity:

                def batched_elbo(X, Y, lik, lik_vars, kernel, kernel_vars, sparsity, sparsity_vars, approximate_posterior, approximate_posterior_vars):

                    N = X.shape[0]
                    Y = Y[:, None]

                    lik.set_vars(lik_vars)
                    kernel.set_vars(kernel_vars)
                    sparsity.set_vars(sparsity_vars)

                    approximate_posterior.set_vars(approximate_posterior_vars)

                    elbo_q = _elbo_q(
                        X, 
                        Y, 
                        lik.get_obj(),
                        kernel.get_obj(),
                        sparsity.get_obj(),
                        approximate_posterior.get_obj()

                    )

                    return elbo_q


                elbo = jax.vmap(batched_elbo, (None, 1, None, 0, None, 0, None, 0, None, 0), 0)(X, Y, likelihood, likelihood.get_vars(), kernel, kernel.get_vars(), sparsity, sparsity.get_vars(), approximate_posterior, approximate_posterior.get_vars() )

                elbo = np.sum(elbo)


        return elbo
