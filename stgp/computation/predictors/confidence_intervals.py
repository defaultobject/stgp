import jax
import jax.numpy as np
import chex

from ...dispatch import evoke, dispatch

# Import types
from ...core import Model
from ...transforms import Transform, LinearTransform, NonLinearTransform, DataLatentPermutation
from ...likelihood import ProductLikelihood
from ...data import Data, TransformedData
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo

@dispatch(Data, Model, ProductLikelihood, DataLatentPermutation)
@dispatch(Data, Model, ProductLikelihood, NonLinearTransform)
def confidence_intervals(XS, m):
    latents = m.prior.latent_obj

    mu =  evoke('marginal', 'samples', m.approximate_posterior, m.likelihood, m.prior)(
        XS, m.data, m.approximate_posterior, m.likelihood, m.prior, m.inference, True
    ) 

    # Ensure correct shape
    mu = np.reshape(mu, [m.inference.prediction_samples, XS.shape[0], m.prior.output_dim])

    ci_lower = jax.numpy.percentile(mu, 2.5, axis=0)
    median = jax.numpy.percentile(mu, 50, axis=0)
    ci_upper = jax.numpy.percentile(mu, 97.5, axis=0)

    # ensure shape is [P, N]
    return median.T, ci_lower.T, ci_upper.T

@dispatch(Data, Model, ProductLikelihood, LinearTransform)
def confidence_intervals(XS, m):
    mu, var = m.predict_y(XS, squeeze=False, diagonal=True)

    P = mu.shape[0]

    # Ensure rank 2
    mu = np.reshape(mu, [P, -1])
    var = np.reshape(var, [P, -1])

    return mu, mu-1.96*np.sqrt(var), mu+1.96*np.sqrt(var)

@dispatch(TransformedData, Model, ProductLikelihood, LinearTransform)
@dispatch(TransformedData, Model, ProductLikelihood, NonLinearTransform)
@dispatch(TransformedData, Model, ProductLikelihood, DataLatentPermutation)
def confidence_intervals(XS, m):
    base_data = m.data.base_data

    median, lower_ci, upper_ci = evoke(
        'confidence_intervals', base_data, m, m.likelihood, m.prior         
    )(XS, m)

    median = m.data.inverse_transform(median.T).T
    lower_ci = m.data.inverse_transform(lower_ci.T).T
    upper_ci = m.data.inverse_transform(upper_ci.T).T

    return median, lower_ci, upper_ci

@dispatch(Model)
def confidence_intervals(XS, m):
    if m.data.minibatch:
        # TODO: minibatching only works when sparsity is used. Assert this.
        m.data.batch()

    return evoke(
        'confidence_intervals', m.data, m, m.likelihood, m.prior         
    )(XS, m)
