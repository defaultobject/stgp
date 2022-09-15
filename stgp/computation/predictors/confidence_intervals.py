import jax
import jax.numpy as np
import chex

from ...dispatch import evoke, dispatch

# Import types
from ...core import Model
from ...transforms import Transform, LinearTransform, NonLinearTransform, DataLatentPermutation, Independent
from ...likelihood import ProductLikelihood, Likelihood
from ...data import Data, TransformedData
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ...core.model_types import get_model_type, LinearModel, NonLinearModel

@dispatch(Data, Model, Likelihood, NonLinearModel)
def confidence_intervals(XS, m):
    # TODO: this is assuming a variational model
    out_block_dim = 1

    mu = evoke('marginal_prediction_samples', m.approximate_posterior, m.likelihood, m.prior, whiten=m.inference.whiten)(
        XS, m.data, m.approximate_posterior, m.likelihood, m.prior, m.inference, out_block_dim, m.inference.whiten
    )

    # Ensure correct shape
    mu = np.reshape(mu, [m.inference.prediction_samples, XS.shape[0], m.prior.output_dim])

    ci_lower = np.percentile(mu, 2.5, axis=0)
    median = np.percentile(mu, 50, axis=0)
    ci_upper = np.percentile(mu, 97.5, axis=0)

    # ensure shape is [P, N]
    return median.T, ci_lower.T, ci_upper.T

@dispatch(Data, Model, Likelihood, LinearModel)
def confidence_intervals(XS, m):
    mu, var = m.predict_y(XS, squeeze=False, diagonal=True)

    P = mu.shape[0]

    # Ensure rank 2
    mu = np.reshape(mu, [P, -1])
    var = np.reshape(var, [P, -1])

    return mu, mu-1.96*np.sqrt(var), mu+1.96*np.sqrt(var)

@dispatch(TransformedData, Model, Likelihood, LinearModel)
@dispatch(TransformedData, Model, Likelihood, NonLinearModel)
def confidence_intervals(XS, m):
    base_data = m.data.base_data

    model_type = get_model_type(m.prior)

    median, lower_ci, upper_ci = evoke(
        'confidence_intervals', base_data, m, m.likelihood, model_type         
    )(XS, m)

    median = m.data.inverse_transform(median.T).T
    lower_ci = m.data.inverse_transform(lower_ci.T).T
    upper_ci = m.data.inverse_transform(upper_ci.T).T

    return median, lower_ci, upper_ci

# =========================== Entry Point  ===========================
@dispatch(Model)
def confidence_intervals(XS, m):
    if m.data.minibatch:
        # TODO: minibatching only works when sparsity is used. Assert this.
        m.data.batch()

    model_type = get_model_type(m.prior)

    return evoke(
        'confidence_intervals', m.data, m, m.likelihood, model_type         
    )(XS, m)
