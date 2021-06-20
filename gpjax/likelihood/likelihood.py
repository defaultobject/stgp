"""Base likelihood class."""

import objax
import jax


class Likelihood(objax.Module):
    """Base likelihood class."""

    def batched_log_likelihood(self, Y, F):
        return jax.vmap(self.log_likelihood, (0, 0), 0)(Y, F)


