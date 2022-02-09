"""Batch inference class."""
from . import Inference
#from ..computation.log_marginal_likelihoods import multi_latent_log_marginal_likelihood, log_marginal_likelihood
#from ..computation.predictors import multi_latent_predict, multi_latent_predictive_covar
from ..dispatch import evoke


class Batch(Inference):
    """Batch inference class."""

    def neg_log_marginal_likelihood(self, X, Y, gp, likelihood, prior):
        lml = evoke('log_marginal_likelihood', gp, likelihood, prior)(
            X, Y, gp, likelihood, prior
        )

        return - lml

    def predict_f(self, XS, X, Y, gp, likelihood, prior, diagonal: bool):

        pred_mu, pred_var = evoke('predict', gp, likelihood, prior)(
            XS, X, Y, gp, likelihood, prior, diagonal
        )

        return pred_mu, pred_var

    def predict_y(self, XS, X, Y, gp, likelihood, prior, diagonal: bool):

        pred_mu, pred_var = predict_f(XS, X, Y, gp, likelihood, prior, diagonal)
        pred_y_mu, pred_y_var = evoke('predict_y', gp, likelihood, prior)(
            XS, gp, likelihood, pred_mu, pred_var, diagonal
        )

        return pred_y_mu, pred_y_var

    def predictive_covar(self, XS_1, XS_2, X, Y, likelihood, prior):

        pred_var = multi_latent_predictive_covar(
            XS_1, XS_2, X, Y, likelihood, prior
        )

        return pred_var
