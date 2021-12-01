"""Variational inference class."""
from . import Inference
from .. import settings

from ..computation.elbos import elbo
from ..computation.predictors import multi_latent_predict

import jax
import jax.numpy as np
import objax
import chex

class Variational(Inference):
    """Variational inference class."""
    def __init__(self, whiten=False, minibatch_size=False):
        super(Variational, self).__init__()

        self.whiten = whiten
        self.minibatch_size = minibatch_size
        self.generator = objax.random.Generator(seed=0)

    def predict_f(self, XS, X, Y, likelihood, prior, approximate_posterior, diagonal):
        mu, var = multi_latent_predict.dispatch(type(prior), type(approximate_posterior))(
            XS, X, Y, likelihood, prior, approximate_posterior, diagonal
        )

        return mu, var


    def ELBO(self, X, Y, likelihood, prior, approximate_posterior):
        """
        Args:
            X: NxD input
            Y: NXP outputs
            likelihood: Array of P likelihoods
            prior: 
            approximate_posterior:  approximate posterior
        """

        val = elbo(
            X, Y, likelihood, prior, approximate_posterior, self
        )

        return val
