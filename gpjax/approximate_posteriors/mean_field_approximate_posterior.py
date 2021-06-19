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

class MeanFieldApproximatePosterior(ApproximatePosterior):
    def __init__(self, prior: 'Node', whiten=False):
        self.prior = prior
        self.whiten = whiten

        self.number_latents = self.prior.number_of_latents()

        #TODO: dim
        self.approx_posteriors = objax.ModuleList([
            GaussianApproximatePosterior(dim=100, whiten=self.whiten)
            for q in range(self.number_latents)
        ])


    def ELL(self, X: np.ndarray, Y: np.ndarray, likelihood: 'Likelihood', transform: 'Transform'):
        if isinstance(transform, LinearTransform):
            return linear_transform_ell(X, Y, likelihood, transform, self.approx_posteriors)


        return 0.0


    def KL(self, X: np.ndarray, transform: 'Transform'):

        latents = transform.latents
        num_latents = len(latents)

        if settings.use_loop_mode:
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

