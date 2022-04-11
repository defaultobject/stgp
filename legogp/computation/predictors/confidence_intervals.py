import jax
import jax.numpy as np
import chex

from ...dispatch import evoke, dispatch

# Import types
from ...transforms import Transform
from ...likelihood import ProductLikelihood
from ...data import Data, TransformedData


@dispatch(Data, 'BatchGP', ProductLikelihood, Transform)
def confidence_intervals(XS, m):
    mu, var = m.predict_y(XS)

    return mu, mu-1.96*np.sqrt(var), mu+1.96*np.sqrt(var)

@dispatch(TransformedData, 'BatchGP', ProductLikelihood, Transform)
def confidence_intervals(XS, m):
    base_data = m.data.base_data

    median, lower_ci, upper_ci = evoke(
        'confidence_intervals', base_data, m, m.likelihood, m.prior         
    )(XS, m)

    median = m.data.inverse_transform(median.T).T
    lower_ci = m.data.inverse_transform(lower_ci.T).T
    upper_ci = m.data.inverse_transform(upper_ci.T).T
    return median, lower_ci, upper_ci

@dispatch('BatchGP')
def confidence_intervals(XS, m):
    return evoke(
        'confidence_intervals', m.data, m, m.likelihood, m.prior         
    )(XS, m)


