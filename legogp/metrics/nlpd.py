""" 
Negative Log Predictive Density (NLPD) computations.

The NLPD is given by:
    NLPD = - log p(Y*_n | X, Y) =  - log ∫ p(Y*_n | F*_n) p(F*_n | X, Y) d F*_n

In the variational settings we approximate p(F*_n | X, Y) with q(F*_n). 

There are three settings:

    Exact computation:
        When the integrals are analytically we compute the NLPD exactly

    Quadrature computation:
        When the integrals are not analytical and the the likelihood decomposes we use quadrature

    Monte-carlo Computation:
        In all other cases we use a monte-carlo estimation.


    In the non-analytical settings we rewrite the NLPD using the Log-Exp trick (for numerical stability):
"""

import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
import chex

from ..dispatch import dispatch, evoke
from ..computation.gaussian import log_gaussian, log_gaussian_scalar

# Import Types
from ..data import Data, TransformedData
from ..likelihood import Likelihood, GaussianProductLikelihood
from ..transforms import Independent, LinearTransform
from ..utils.nan_utils import get_same_shape_mask

@dispatch(Data, 'BatchGP', GaussianProductLikelihood, LinearTransform)
@dispatch(Data, 'BatchGP', GaussianProductLikelihood, Independent)
def nlpd(XS, YS, m):
    # Available in closed form
    pred_mu, pred_var = m.predict_y(XS, diagonal=True, squeeze=False)

    chex.assert_rank([YS, pred_mu, pred_var], [2, 2, 2])
    chex.assert_equal_shape([YS, pred_mu.T, pred_var.T])

    mask = get_same_shape_mask(YS)
    Y_masked = np.nan_to_num(YS, nan=0.0)

    res = jax.vmap(
        jax.vmap(log_gaussian_scalar, [0, 0, 0]),
        [0, 1, 1]
    )(Y_masked, pred_mu, pred_var)

    # mask res
    res = res * mask

    # Average over N, ignoring the missing data
    return - np.sum(res, axis=0) / np.sum(mask, axis=0)

@dispatch(TransformedData, 'BatchGP', GaussianProductLikelihood, LinearTransform)
@dispatch(TransformedData, 'BatchGP', GaussianProductLikelihood, Independent)
def nlpd(XS, YS, model):
    """
    In the transformed setting the NLPD is given as:

         - (1/N) \sum^N_n [ \log p(T(YS_n)) + log |dT(YS_n) / d YS_2| ]
    """
    data = model.data
    base_data = data.base_data

    base_nlpd = evoke('nlpd', base_data, model, model.likelihood, model.prior)(
        XS,
        data.forward_transform(YS),
        model
    )
    
    mask = get_same_shape_mask(YS)

    log_jac = data.log_jacobian(YS)
    log_jac = np.nan_to_num(log_jac, 0.0)

    # Average over N, ignoring the missing data
    log_jac = np.sum(log_jac, axis=0) / np.sum(mask, axis=0)

    return base_nlpd - log_jac


@dispatch('BatchGP')
def nlpd(XS, YS, model):
    return evoke('nlpd', model.data, model, model.likelihood, model.prior)(
        XS,
        YS,
        model
    )


def nlpd(XS, YS, model):
    return evoke('nlpd', model)( XS, YS, model)

