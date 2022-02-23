"""Base likelihood class."""

import objax
import jax


class Likelihood(objax.Module):
    """Base likelihood class."""

    def log_likelihood(self, Y, F):
        return jax.vmap(self.log_likelihood_scalar, (0, 0), 0)(Y, F)

class DiagonalLikelihood(Likelihood):
    """Likelihood that decomposes across data """
