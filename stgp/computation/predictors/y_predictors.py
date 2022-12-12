import jax
from jax import jit
import jax.numpy as np
import chex

from ...likelihood import ProductLikelihood
from ...transforms import LinearTransform, Independent, Transform
from ...dispatch import dispatch, evoke
from ...utils.batch_utils import batch_over_module_types
from batchjax import batch_or_loop, BatchType
from ..matrix_ops import add_jitter, vec_add_jitter
from ..model_ops import get_vec_gaussian_likelihood_variances
from ...core import Posterior

# Gaussian Likelihoods

# ======= Full var ========
@dispatch(Posterior, 'Gaussian')
def predict_y_full(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [2, 3])
    breakpoint()
    return post_mu, add_jitter(post_var, likelihood.variance)

@dispatch(Posterior, 'GaussianProductLikelihood')
def predict_y_full(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [2, 3])
    N, P = post_mu.shape

    lik_var = np.diag(likelihood.variance)
    chex.assert_shape(lik_var, [P, P])

    return post_mu, vec_add_jitter(post_var, lik_var)

# ======= Diagonal var ========

@dispatch(Posterior, 'Gaussian')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [2, 3])
    chex.assert_equal([post_var.shape[1], post_var.shape[2]], [1, 1])

    return post_mu, post_var + likelihood.variance

@dispatch(Posterior, 'ReshapedGaussian')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [2, 3])
    chex.assert_equal([post_var.shape[1], post_var.shape[2]], [1, 1])

    return post_mu, post_var + likelihood.base.variance

# ======= Dispatchers ========

@dispatch(Posterior, "HetGaussian", Transform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):
    if diagonal:
        m_f = post_mu[:, 0, ...][:, None, ...]
        m_g = post_mu[:, 1, ...][:, None, ...]

        k_f = post_var[:, 0, ...]
        k_g = post_var[:, 1, ...]

        mean = m_f
        var = k_f + np.exp(2 * m_g + 2 * k_g)

        return mean, var[..., None]

@dispatch(Posterior, ProductLikelihood, Transform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):

    chex.assert_rank([post_mu, post_var], [3, 4])

    if diagonal:
        likelihood_arr = likelihood.likelihood_arr
        num_outputs = len(likelihood_arr)

        # Compute prediction for each likelihood-prior pair
        mu_arr, var_arr =  batch_over_module_types(
            'predict_y_diagonal',
            [gp],
            likelihood_arr,
            [XS, likelihood_arr, post_mu, post_var],
            [None, 0, 1, 1],
            num_outputs,
            2
        )

        # fix data-latent ordering due to batching
        mu_arr = np.transpose(mu_arr, [1, 0, 2])
        var_arr = np.transpose(var_arr, [1, 0, 2, 3])

        return mu_arr, var_arr

    else:
        # TODO: this will break with aggreagtion
        # fix shapes
        post_mu = post_mu[..., 0]
        post_var = post_var[:, 0, ...]

        mu, var =  evoke('predict_y_full', gp, likelihood)(
            XS, likelihood, post_mu, post_var
        )

        # add back removed dimensions
        mu = mu[..., None]
        var = var[:, None, ...]

        chex.assert_rank([mu, var], [3, 4])

        return mu, var

@dispatch('BatchGP', 'ReshapedGaussian', LinearTransform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):
    if diagonal:
        lik_var = get_vec_gaussian_likelihood_variances(
            np.transpose(post_var, [1, 0]),
            likelihood_arr
        )

        return post_mu, post_var+lik_var
    else:
        raise NotImplementedError()


@dispatch('BatchGP', 'GaussianProductLikelihood', LinearTransform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):

    if diagonal:
        chex.assert_rank([post_mu, post_var], [2, 2])
        chex.assert_equal_shape([post_mu, post_var])

        lik_var = likelihood.base.variance

        chex.assert_rank(lik_var, 1)

        return post_mu, post_var + lik_var[:, None]
    else:
        raise NotImplementedError()

