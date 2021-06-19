import jax
import jax.numpy as np
import objax
import chex
from .. import settings

from ..transform  import LinearTransform, NonLinearTransform
from . import ApproximatePosterior, GaussianApproximatePosterior
from ..batching import batch
from .. batching import Batched
from ..computation.matrix_ops import vectorized_lower_triangular_cholesky, lower_triangle
import chex
from ..computation.ell_callers import linear_transform_ell
from ..computation.predictor_callers import linear_predictor

class MeanFieldApproximatePosterior(ApproximatePosterior):
    def __init__(self, prior: 'Node', whiten=False):
        self.whiten = whiten

        self.number_latents = prior.number_of_latents()

        #TODO: dim
        self.approx_posteriors = objax.ModuleList([
            GaussianApproximatePosterior(dim=100, whiten=self.whiten)
            for q in range(self.number_latents)
        ])

    def predict(self, XS, X, likelihood, prior, diagonal):
        return linear_predictor(XS, X, likelihood, prior, self)

    def precompute_marginals(self, X, prior):
        latents = prior.latents
        num_latents = len(latents)

        with Batched(latents) as latents, Batched(self.approx_posteriors) as approx_posteriors:
            def batched_marginals(X, latent, latent_vars, approx_posterior, approx_posterior_vars):
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

                return mean_q,  var_q 

            mean_arr, var_arr = jax.vmap(batched_marginals, (None,  None, 0, None, 0), (0, 0))(X, latents, latents.get_vars(), approx_posteriors, approx_posteriors.get_vars())


            chex.assert_equal(mean_arr.shape, (num_latents, X.shape[0], 1))
            chex.assert_equal(var_arr.shape, (num_latents, X.shape[0], 1))


        self.precomputed_marginal_mean_arr = mean_arr
        self.precomputed_marginal_var_arr = var_arr

    def ELL(self, X: np.ndarray, Y: np.ndarray, likelihood: 'Likelihood', transform: 'Transform'):
        if isinstance(transform, LinearTransform):
            return linear_transform_ell(X, Y, likelihood, transform, self)

        return 0.0

    def KL(self, X: np.ndarray, transform: 'Transform'):
        latents = transform.latents
        num_latents = len(latents)

        if False and settings.use_loop_mode:
            raise NotImplementedError()
        else:

            with Batched(latents) as latents, Batched(self.approx_posteriors) as approx_posteriors:
                def batched_kl(X, latent, latent_vars, approx_posterior, approx_posterior_vars):
                    latent.set_vars(latent_vars)
                    approx_posterior.set_vars(approx_posterior_vars)

                    latent = latent.get_obj()
                    approx_posterior = approx_posterior.get_obj()

                    return approx_posterior.KL(
                        X, 
                        latent.kernel,
                        latent.sparsity
                    )

                kl = jax.vmap(batched_kl, (None,  None, 0, None, 0), 0)(X, latents, latents.get_vars(),  approx_posteriors, approx_posteriors.get_vars())

            chex.assert_equal(kl.shape, (num_latents, ))

        return np.sum(kl)

