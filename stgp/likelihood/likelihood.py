"""Base likelihood class."""

import objax
import jax
import chex


class Likelihood(objax.Module):
    """Base likelihood class."""

    @property
    def block_size(self):
        return 1

    @property
    def base(self):
        return self

    def log_likelihood(self, Y, F):
        chex.assert_shape(Y, F.shape)
        chex.assert_rank(Y, 2)
        chex.assert_equal(Y.shape[1], 1)

        return jax.vmap(self.log_likelihood_scalar, (0, 0), 0)(Y[:, 0], F[:, 0])

    def conditional_var(self, f):
        raise NotImplementedError()

    def conditional_mean(self, f):
        raise NotImplementedError()

class FullLikelihood(Likelihood):
    """Likelihood that does not decompose """

class DiagonalLikelihood(Likelihood):
    """Likelihood that decomposes across data """

class BlockDiagonalLikelihood(Likelihood):
    """Likelihood that can decompose across blocks """
