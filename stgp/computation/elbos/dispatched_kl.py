""" KL dispatchers """
import chex
import jax
import jax.numpy as np

from ...dispatch import dispatch, evoke
from .kullback_leiblers import gaussian_cholesky_kl
from ...transforms import Transform, Independent
from ...utils.batch_utils import batch_over_module_types
from ..matrix_ops import cholesky, add_jitter
from ... import settings
from .prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ, prior_mean_Z, prior_mean_X

from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior, FullGaussianApproximatePosterior
from ...core import GPPrior
from ...transforms import DataLatentPermutation, Joint

@dispatch(GaussianApproximatePosterior, Joint)
def kullback_leibler(approximate_posterior, prior):
    """ Compute KL between Gaussian approximate posterior and Gaussian prior"""
    Z = prior.get_Z()

    covar_2 = prior.b_covar(Z, Z)

    covar_chol_2 = cholesky(add_jitter(covar_2, settings.jitter))

    return gaussian_cholesky_kl(
        approximate_posterior.m,
        approximate_posterior.S_chol,
        prior.b_mean(Z),
        covar_chol_2
    )

@dispatch(GaussianApproximatePosterior, GPPrior)
def kullback_leibler(approximate_posterior, prior):
    """ Compute KL between Gaussian approximate posterior and Gaussian prior"""
    Z = prior.get_Z()

    covar_2 = prior.covar(Z, Z)

    covar_chol_2 = cholesky(add_jitter(covar_2, settings.jitter))

    return gaussian_cholesky_kl(
        approximate_posterior.m,
        approximate_posterior.S_chol,
        prior.mean(Z),
        covar_chol_2
    )

@dispatch(MeanFieldApproximatePosterior, Independent)
def kullback_leibler(approximate_posterior, prior):
    """ Compute KL between mean-field Gaussian approximate posterior and mean-field Gaussian prior"""

    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors

    kl_arr = batch_over_module_types(
        evoke_name = 'kullback_leibler',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, latents_arr],
        fn_params = [approx_posteriors_arr, latents_arr],
        fn_axes = [0, 0],
        dim = len(latents_arr),
        out_dim  = 1 
    )

    chex.assert_shape(kl_arr, [len(latents_arr)])

    return np.sum(kl_arr)


@dispatch(MeanFieldApproximatePosterior, Transform)
def kullback_leibler(approximate_posterior, prior):
    """
    A prior is constructed by defining joint/independent GPs and then transforming them. 
    The KL term is only calculated on the GPs not the transformed ones.
    """

    latents = prior.base_prior

    return evoke('kullback_leibler', approximate_posterior, latents)(
        approximate_posterior, latents
    )

@dispatch(FullGaussianApproximatePosterior, Transform)
def kullback_leibler(approximate_posterior, prior):
    """
    Both the prior and the approximate psoterior are defined in latent-data format.
    """
    # use base prior as that is where the latent GPs are defined
    base_prior = prior.base_prior

    return evoke('kullback_leibler', approximate_posterior, base_prior)(
        approximate_posterior, base_prior
    )


