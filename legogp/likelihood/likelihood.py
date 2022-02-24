"""Base likelihood class."""

import objax
import jax
import chex


class Likelihood(objax.Module):
    """Base likelihood class."""

    def log_likelihood(self, Y, F):
        chex.assert_shape(Y, F.shape)
        chex.assert_rank(Y, 2)
        chex.assert_equal(Y.shape[1], 1)

        return jax.vmap(self.log_likelihood_scalar, (0, 0), 0)(Y[:, 0], F[:, 0])

class DiagonalLikelihood(Likelihood):
    """Likelihood that decomposes across data """
