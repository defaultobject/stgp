"""Variational inference class."""
from . import Inference
from .. import settings
from ..dispatch import evoke

from ..computation.elbos import *

import jax
import jax.numpy as np
import objax
import chex

class Variational(Inference):
    """Variational inference class."""
    def __init__(self, whiten=False, minibatch_size=False, ell_samples=1, prediction_samples=100):
        super(Variational, self).__init__()

        self.whiten = whiten
        self.minibatch_size = minibatch_size
        self.generator = objax.random.Generator(seed=0)
        self.ell_samples=ell_samples
        self.prediction_samples=prediction_samples

    def predict_f(self, XS, X, Y, likelihood, prior, approximate_posterior, diagonal):
        mu, var = evoke('predict', likelihood, prior, approximate_posterior)(
            XS, X, Y, likelihood, prior, approximate_posterior, self, diagonal
        )

        return mu, var

    def predict_latents(self, XS, X, Y, likelihood, prior, approximate_posterior, diagonal):

        latents = prior.latents

        return evoke('marginal', 'prediction', approximate_posterior, latents)(
            XS, X, approximate_posterior, latents
        )

    def predict_y(self, XS, X, Y, likelihood, prior, approximate_posterior, diagonal):
        return self.predict_f(XS, X, Y, likelihood, prior, approximate_posterior, diagonal)


    def predictive_covar(self, XS_1, XS_2, X, Y, likelihood, prior, approximate_posterior):
        pred_var = multi_latent_predictive_covar.dispatch(type(prior), type(approximate_posterior))(
            XS_1, XS_2, X, Y, likelihood, prior, approximate_posterior

        )
        return pred_var


    def ELBO(self, X, Y, likelihood, prior, approximate_posterior):
        """
        Args:
            X: NxD input
            Y: NXP outputs
            likelihood: Array of P likelihoods
            prior: 
            approximate_posterior:  approximate posterior
        """

        val = evoke('elbo', likelihood, prior, approximate_posterior)(
            X, Y, likelihood, prior, approximate_posterior, self
        )

        return val
