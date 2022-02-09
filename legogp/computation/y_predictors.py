import jax
from jax import jit
import jax.numpy as np
import chex

from ..dispatch import dispatch, evoke
from ..utils.utils import can_batch, get_batch_type
from batchjax import batch_or_loop, BatchType
from .matrix_ops import add_jitter

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

    # if all likelihooods are the same we only need the first object
    #   and then we can batch it
    # otherwises we need the whole array and we will loop through them all
    if can_batch(likelihood_arr):
        pred_fn = evoke(evoke_name, gp, likelihood_arr[0])
        pred_axes = None
    else:
        pred_fn = [evoke(evoke_name, gp, lik) for lik in likelihood_arr]
        pred_axes = 0

    fn = lambda pred_fn, *args: pred_fn(*args)
    # Compute prediction for each likelihood-prior pair
    mu_arr, var_arr = batch_or_loop(
        fn,
        [pred_fn, XS, likelihood_arr, post_mu, post_var],
        [pred_axes, None, 0, 0, 0],
        dim=num_outputs,
        out_dim=2,
        batch_type = get_batch_type(likelihood_arr)
    )

    return mu_arr, var_arr
