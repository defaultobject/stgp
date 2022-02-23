import chex
import jax
import jax.numpy as np

from ...dispatch import dispatch, evoke
from ...transforms import LinearTransform, Independent
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import get_mask, mask_vector
from ...likelihood import ProductLikelihood, DiagonalLikelihood
from .expected_log_likelihoods import scalar_gaussian_expected_log_likelihood, gaussian_expected_log_likelihood

@dispatch('scalar', 'Gaussian', 'GaussianApproximatePosterior')
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    return scalar_gaussian_expected_log_likelihood(X, Y, likelihood.variance, q_f_mu, q_f_var)

@dispatch(DiagonalLikelihood, 'GaussianApproximatePosterior')
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ For diagonal likelihoods adds support for missing data. """ 

    # Get nan mask for output
    mask = get_mask(Y)

    # Convert nans to zeros
    Y = mask_vector(Y, mask)

    X = X[..., None]
    Y = Y[..., None]
    q_f_mu = q_f_mu[..., None]
    q_f_var = q_f_var[..., None]

    fn = evoke('expected_log_likelihood', 'scalar', likelihood, 'GaussianApproximatePosterior')

    # Compute ELL for each datapoint
    ell_arr = jax.vmap(
        fn,
        [0, 0, 0, 0, None],
        0
    )(X, Y, q_f_mu, q_f_var, likelihood)

    # Set elements that correposnd to missing data to zero
    ell_arr = mask_vector(ell_arr[:, None], mask)

    # Only sums the ELL terms without missing data
    ell = np.sum(ell_arr)

    return ell

@dispatch('GaussianApproximatePosterior', False)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, approx_posterior, minibatch):
    return evoke('expected_log_likelihood', likelihood, approx_posterior)(
        X, Y, q_f_mu, q_f_var, likelihood
    )


@dispatch(ProductLikelihood, LinearTransform, 'MeanFieldApproximatePosterior')
def expected_log_likelihood(X, Y, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior):
    """
    When the prior is a linear transform the approximate posterior is Gaussian and 
        q_f_mu, q_f_var will already be transformed if necessary
    """

    likelihood_arr = likelihood.likelihood_arr
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    minibatch_arr = [False, False]

    Y = Y[..., None]

    ell_arr = batch_over_module_types(
        evoke_name = 'expected_log_likelihood',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, minibatch_arr],
        fn_params = [X, Y, q_f_mu_arr, q_f_var_arr, likelihood_arr, approx_posteriors_arr, minibatch_arr],
        fn_axes = [None, 1, 0, 0, 0, 0, 0],
        dim = len(approx_posteriors_arr),
        out_dim  = 1 
    )

    chex.assert_shape(ell_arr, [len(likelihood_arr)])

    return np.sum(ell_arr)
