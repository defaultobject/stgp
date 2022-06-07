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

    def predict_f(self, XS, data, likelihood, prior, approximate_posterior, diagonal):

        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        mu, var = evoke('predict', likelihood, prior, approximate_posterior)(
            XS, data, likelihood, prior, approximate_posterior, self, diagonal
        )

        return mu, var

    def predict_latents(self, XS, data, likelihood, prior, approximate_posterior, diagonal):
        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        return evoke('marginal', 'latents', approximate_posterior, likelihood, prior)(
            XS, data, approximate_posterior, likelihood, prior, self, diagonal
        )

    def predict_y(self, XS, data, likelihood, prior, approximate_posterior, diagonal):
        pred_mu, pred_var = self.predict_f(XS, data, likelihood, prior, approximate_posterior, diagonal)

        if True:
            pred_y_mu, pred_y_var = evoke('predict_y', approximate_posterior, likelihood, prior)(
                XS, approximate_posterior, likelihood, pred_mu, pred_var, diagonal
            )

            return pred_y_mu, pred_y_var

        return pred_mu, pred_var


    def predictive_covar(self, XS_1, XS_2, data, likelihood, prior, approximate_posterior):
        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        pred_var = multi_latent_predictive_covar.dispatch(type(prior), type(approximate_posterior))(
            XS_1, XS_2, data, likelihood, prior, approximate_posterior

        )
        return pred_var


    def ELBO(self, data, likelihood, prior, approximate_posterior):
        """
        Args:
            data: Data
            likelihood: Array of P likelihoods
            prior: 
            approximate_posterior:  approximate posterior
        """

        val = evoke('elbo', likelihood, prior, approximate_posterior)(
            data, likelihood, prior, approximate_posterior, self
        )

        return val
