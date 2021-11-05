"""Batch inference class."""
from . import Inference
from ..computation.log_marginal_likelihoods import multi_latent_log_marginal_likelihood
from ..computation.predictors import multi_latent_predict, multi_latent_predictive_covar


class Batch(Inference):
    """Batch inference class."""

    def neg_log_marginal_likelihood(self, X, Y, likelihood, prior):
        lml = multi_latent_log_marginal_likelihood(
            X, Y, likelihood, prior
        )

        return - lml

    def predict(self, XS, X, Y, likelihood, prior, diagonal: bool):

        pred_mu, pred_var = multi_latent_predict(
            XS, X, Y, likelihood, prior, diagonal
        )

        # ensure correct shapes
        return pred_mu, pred_var

    def predictive_covar(self, XS_1, XS_2, X, Y, likelihood, prior):

        pred_var = multi_latent_predictive_covar(
            XS_1, XS_2, X, Y, likelihood, prior
        )

        return pred_var
