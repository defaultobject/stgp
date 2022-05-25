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
from batchjax import batch_or_loop, BatchType

from ..dispatch import dispatch, evoke
from ..computation.gaussian import log_gaussian, log_gaussian_scalar
from ..utils.nan_utils import get_same_shape_mask
from ..utils.utils import get_batch_type

# Import Types
from ..core import Model
from ..data import Data, TransformedData
from ..likelihood import Likelihood, GaussianProductLikelihood, ProductLikelihood
from ..transforms import Independent, LinearTransform, Transform, NonLinearTransform, DataLatentPermutation
from ..inference import Inference, Variational, Batch


@dispatch(Data, 'VGP', ProductLikelihood, NonLinearTransform, 'Variational')
def nlpd(XS, YS, m, prior):
    """
    For each n:
        NLPD = - log p(Y*_n | X, Y)
            \approx -(1\S) \sum^S_s p(Y*_n | F^(s)_n) for F^(s)_n \sim q(F_n)

        and we compute inner sum using the log sum exp trick
        
    """

    mask = get_same_shape_mask(YS)
    Y_masked = np.nan_to_num(YS, nan=0.0)


    #compute samples
    # shape will be [n_samples, N, P, 1] or [n_samples, N, P]
    m_samples = evoke('marginal', 'samples', m.approximate_posterior, m.likelihood, prior)(
        XS, m.data, m.approximate_posterior, m.likelihood, m.prior, m.inference, True
    )

    P = YS.shape[1]
    N = XS.shape[0]
    n_samples = m_samples.shape[0]

    # ensure consistent shape
    m_samples = np.reshape(m_samples, [n_samples, N, P])

    lik_arr = m.likelihood.likelihood_arr

    ll_arr = batch_or_loop(
        lambda f_ns, y_p, lik: jax.vmap(lambda f, y: lik.log_likelihood(f[:, None], y[:, None]), [0, None])(f_ns, y_p),
        [m_samples, Y_masked, lik_arr],
        [2, 1, 0],
        dim = len(lik_arr),
        out_dim=1,
        batch_type = get_batch_type(lik_arr)
    )
    # ll_arr will have shape [P, n_samples, N]
    # average over n_samples for each N
    ll_arr = ll_arr

    # vmap over P
    res = jax.vmap(
        lambda ll: jax.vmap(lambda l: logsumexp(l), [1])(ll), #vmap over N
        [0]
    )(ll_arr)

    res = res + np.log(1/n_samples)

    chex.assert_shape(res, [P, N])

    # N x P
    res = res.T

    # mask res
    res = res * mask

    # Average over N, ignoring the missing data
    return - np.sum(res, axis=0) / np.sum(mask, axis=0)

@dispatch(Data, 'VGP', GaussianProductLikelihood, LinearTransform, 'Variational')
def nlpd(XS, YS, model, prior):
    """ With a gaussian likelihood this is exact. """

    return evoke('nlpd', model.data, model, model.likelihood, prior, Inference)(
        XS,
        YS,
        model,
        prior
    )

@dispatch(Data, Model, ProductLikelihood, DataLatentPermutation, 'Variational')
def nlpd(XS, YS, m, prior):
    base_prior = prior.latent_obj

    return evoke('nlpd', m.data, m, m.likelihood, base_prior, m.inference)(
        XS,
        YS,
        m,
        base_prior
    )

@dispatch(Data, Model, GaussianProductLikelihood, LinearTransform, Inference)
@dispatch(Data, Model, GaussianProductLikelihood, Independent, Inference)
def nlpd(XS, YS, m, prior):
    """ Closed form Gaussian NLPD """
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

@dispatch(TransformedData, Model, Likelihood, Transform, Inference)
def nlpd(XS, YS, model, prior):
    """
    In the transformed setting the NLPD is given as:

         - (1/N) \sum^N_n [ \log p(T(YS_n)) + log |dT(YS_n) / d YS_2| ]
    """
    data = model.data
    base_data = data.base_data

    base_nlpd = evoke('nlpd', base_data, model, model.likelihood, model.prior, model.inference)(
        XS,
        data.forward_transform(YS),
        model,
        model.prior
    )
    
    mask = get_same_shape_mask(YS)

    log_jac = data.log_jacobian(YS)
    log_jac = np.nan_to_num(log_jac, 0.0)

    # Average over N, ignoring the missing data
    log_jac = np.sum(log_jac, axis=0) / np.sum(mask, axis=0)

    return base_nlpd - log_jac


@dispatch('BatchGP')
def nlpd(XS, YS, model):
    return evoke('nlpd', model.data, model, model.likelihood, model.prior, model.inference)(
        XS,
        YS,
        model,
        model.prior
    )

@dispatch('VGP')
def nlpd(XS, YS, model):
    """ 
    In the variational setting the NLPD is approximated as:
    NLPD = - log p(Y*_n | X, Y) =  - log ∫ p(Y*_n | F*_n) q(F*_n) d F*_n
    """
    return evoke('nlpd', model.data, model, model.likelihood, model.prior, model.inference)(
        XS,
        YS,
        model,
        model.prior
    )

def nlpd(XS, YS, model):
    return evoke('nlpd', model)( XS, YS, model)

