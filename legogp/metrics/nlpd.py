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
from ..likelihood import Likelihood, GaussianProductLikelihood
from ..transforms import Independent, LinearTransform

@dispatch('BatchGP', GaussianProductLikelihood, LinearTransform)
@dispatch('BatchGP', GaussianProductLikelihood, Independent)
def nlpd(XS, YS, m):
    # Available in closed form
    pred_mu, pred_var = m.predict_y(XS, diagonal=True, squeeze=False)

    chex.assert_rank([YS, pred_mu, pred_var], [2, 2, 2])
    chex.assert_equal_shape([YS, pred_mu.T, pred_var.T])

    res = jax.vmap(
        jax.vmap(log_gaussian_scalar, [0, 0, 0]),
        [0, 1, 1]
    )(YS, pred_mu, pred_var)

    # Sum over P, Average over N
    return - np.mean(np.sum(res, axis=1), axis=0)

@dispatch('BatchGP')
def nlpd(XS, YS, model):
    return evoke('nlpd', model, model.likelihood, model.prior)(
        XS,
        YS,
        model
    )


def nlpd(XS, YS, model):
    return evoke('nlpd', model)( XS, YS, model)

