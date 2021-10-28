"""Batch inference class."""
from . import Inference
from ..computation.log_marginal_likelihoods import multi_latent_log_marginal_likelihood
from ..computation.predictors import multi_latent_predict


class Batch(Inference):
    """Batch inference class."""

    def neg_log_marginal_likelihood(self, X, Y, likelihood, prior, mask):
        lml = multi_latent_log_marginal_likelihood(
            X, Y, likelihood, prior, mask
        )

        return - lml

    def predict(self, XS, X, Y, likelihood, prior, mask, diagonal: bool):

        pred_mu, pred_var = multi_latent_predict(
            XS, X, Y, likelihood, prior, mask, diagonal
        )

        # ensure correct shapes

        return pred_mu, pred_var
