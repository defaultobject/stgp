import jax
from jax import jit
import jax.numpy as np
import chex

from ...dispatch import dispatch, evoke
from ...utils.batch_utils import batch_over_likelihoods
from batchjax import batch_or_loop, BatchType
from ..matrix_ops import add_jitter

# Gaussian Likelihoods
@dispatch('BatchGP', 'Gaussian')
def predict_y_full(XS, likelihood, post_mu, post_var):

    return post_mu, add_jitter(post_var, likelihood.variance)

@dispatch('BatchGP', 'Gaussian')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):

    return post_mu, post_var + likelihood.variance


@dispatch('BatchGP', 'ProductLikelihood', 'Independent')
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):
    num_outputs = gp.output_dim

    if diagonal:
        evoke_name = 'predict_y_diagonal'
    else:
        evoke_name = 'predict_y_full'

    likelihood_arr = likelihood.likelihood_arr

    # Compute prediction for each likelihood-prior pair
    mu_arr, var_arr =  batch_over_likelihoods(
        evoke_name,
        [gp],
        likelihood_arr,
        [XS, likelihood_arr, post_mu, post_var],
        [None, 0, 0, 0],
        num_outputs,
        2
    )

    return mu_arr, var_arr
