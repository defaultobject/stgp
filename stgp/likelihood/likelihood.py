"""Base likelihood class."""

import objax
import jax
import jax.numpy as np
import chex

from ..core import Block


class Likelihood(objax.Module):
    """Base likelihood class."""

    @property
    def block_type(self):
        raise NotImplementedError()

    @property
    def base(self):
        return self

    def log_likelihood(self, Y, F):
        chex.assert_equal(Y.shape[0], F.shape[0])

        return jax.vmap(self.log_likelihood_scalar, (0, 0), 0)(
            np.squeeze(Y), 
            np.squeeze(F)
        )

    def conditional_var(self, f):
        raise NotImplementedError()

    def conditional_mean(self, f):
        raise NotImplementedError()

class FullLikelihood(Likelihood):
    """Likelihood that does not decompose """
    @property
    def block_type(self):
        return Block.FULL

class DiagonalLikelihood(Likelihood):
    """Likelihood that decomposes across data """
    @property
    def block_type(self):
        return Block.DIAGONAL

class BlockDiagonalLikelihood(Likelihood):
    """Likelihood that can decompose across blocks """
    @property
    def block_type(self):
        return Block.BLOCK
