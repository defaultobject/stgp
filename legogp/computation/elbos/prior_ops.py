""" Helper functions for batching in variational models """
import jax
import objax
import chex

from batchjax import batch_or_loop
from ...utils.utils import can_batch

def _batch_over_prior(prior, fn):
    return batch_or_loop(
        fn,
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_flag = can_batch(prior.latents)
    )

def prior_mean_Z(prior):
    K_zz_arr = _batch_over_prior(
        prior,
        lambda prior: prior.kernel.K(prior.sparsity.Z, prior.sparsity.Z),
    )
    return K_zz_arr

def prior_covar_ZZ(prior):
    K_zz_arr = _batch_over_prior(
        prior,
        lambda prior: prior.kernel.K(prior.sparsity.Z, prior.sparsity.Z),
    )
    return K_zz_arr

def prior_covar_XZ(prior, X):
    K_xz_arr = _batch_over_prior(
        prior,
        lambda prior: prior.kernel.K(X, prior.sparsity.Z),
    )
    return K_xz_arr
