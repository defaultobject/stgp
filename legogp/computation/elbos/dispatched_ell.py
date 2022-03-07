import chex
import jax
import jax.numpy as np
import objax

from ...dispatch import dispatch, evoke
from ...transforms import LinearTransform, Independent, NonLinearTransform, Transform
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import get_mask, mask_vector, mask_matrix, get_same_shape_mask
from ...utils.utils import get_batch_type
from ...likelihood import ProductLikelihood, DiagonalLikelihood, Likelihood, DiagonalGaussian, Gaussian, BlockDiagonalGaussian
from .expected_log_likelihoods import scalar_gaussian_expected_log_likelihood, gaussian_expected_log_likelihood, full_gaussian_expected_log_likelihood
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ...approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior


from batchjax import batch_or_loop, BatchType
from numpy.polynomial.hermite import hermgauss

@dispatch('scalar', Gaussian, GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    return scalar_gaussian_expected_log_likelihood(X, Y, likelihood.variance, q_f_mu, q_f_var)


@dispatch('scalar', Likelihood, GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    num_quad_points = 10

    x, w = hermgauss(num_quad_points)
    const = np.pi**-0.5

    q_f_var = np.squeeze(q_f_var)
    q_f_mu = np.squeeze(q_f_mu)
    Y = np.squeeze(Y)

    # change of variable
    f = 2.0**0.5*np.sqrt(q_f_var)*x + q_f_mu  

    chex.assert_shape(f, [num_quad_points])

    res = jax.vmap(
        likelihood.log_likelihood_scalar, 
        [None, 0], 
        0
    )(Y, f)

    chex.assert_shape(res, [num_quad_points])
    
    return np.sum(w * const*res)

@dispatch("DiagonalGaussian", GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ Special case for diagonal Gaussian"""
    N = X.shape[0]

    # Get nan mask for output
    mask = get_mask(Y)

    # Convert nans to zeros
    Y = mask_vector(Y, mask)

    X = X[..., None]
    Y = Y[..., None]
    q_f_mu = q_f_mu[..., None]
    q_f_var = q_f_var[..., None]

    lik_var = likelihood.variance
    chex.assert_shape(lik_var, [N])

    ell_arr = jax.vmap(
        scalar_gaussian_expected_log_likelihood,
        [0, 0, 0, 0, 0],
        0
    )(X, Y, lik_var, q_f_mu, q_f_var)
    chex.assert_shape(ell_arr, [N])

    # Set elements that correposnd to missing data to zero
    ell_arr = mask_vector(ell_arr[:, None], mask)

    # Only sums the ELL terms without missing data
    ell = np.sum(ell_arr)

    return ell


@dispatch(DiagonalLikelihood, GaussianApproximatePosterior)
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

@dispatch(GaussianApproximatePosterior, False)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, approx_posterior, minibatch):
    return evoke('expected_log_likelihood', likelihood, approx_posterior)(
        X, Y, q_f_mu, q_f_var, likelihood
    )

@dispatch(ProductLikelihood, LinearTransform, MeanFieldApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    When the prior is a linear transform the approximate posterior is Gaussian and 
        q_f_mu, q_f_var will already be transformed if necessary
    """

    likelihood_arr = likelihood.likelihood_arr
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    minibatch_arr = [False] * len(approx_posteriors_arr)

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

def compute_ell_for_sample(f, X, Y, prior, likelihood, approx_posteriors_arr):
    chex.assert_rank(f, 2)
    chex.assert_rank(Y, 2)

    likelihood_arr = likelihood.likelihood_arr
    num_likelihoods = len(likelihood_arr)

    # Reparameterise
    # Transform through prior
    transformed_f = jax.vmap(
        prior.forward,
        [1],
        0
    )(f)

    # Y and F must be rank 2 when they are passed to log_likelihood
    # When vmapping one dimension is lost so extent here
    Y = Y[..., None]
    transformed_f = transformed_f[..., None]
    chex.assert_shape(transformed_f, Y.shape)

    # Get nan mask for output
    mask = get_same_shape_mask(Y)

    # Convert nans to zeros
    Y = mask_matrix(Y, mask)

    # batch over outputs
    # log likelihood for each outout
    ll_arr = batch_or_loop(
        lambda y, f, lik: lik.log_likelihood(y, f),
        [Y, transformed_f, likelihood_arr],
        [1, 1, 0],
        dim = num_likelihoods,
        out_dim=1,
        batch_type = get_batch_type(likelihood_arr)
    )

    # Fix shapes so that ll_arr matches Y
    ll_arr = ll_arr[..., None]
    ll_arr = np.transpose(ll_arr, [1, 0, 2])

    chex.assert_equal(ll_arr.shape, Y.shape)

    # Mask out log-liklihoods that correspond to missing data
    ll_arr = mask_matrix(ll_arr, mask)

    return np.sum(ll_arr)

@dispatch(ProductLikelihood, NonLinearTransform, MeanFieldApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Samples from the approximate posteriors need to be transformed through the prior and then the 
        ELL is approximated using monte-carlo
    """

    likelihood_arr = likelihood.likelihood_arr
    approx_posteriors_arr = approximate_posterior.approx_posteriors

    num_likelihoods = len(likelihood_arr)
    N = Y.shape[0]

    return mv_indepentdent_monte_carlo(
        compute_ell_for_sample, 
        q_f_mu_arr, 
        q_f_var_arr, 
        fn_args = [X, Y, prior, likelihood, approx_posteriors_arr],
        generator = inference.generator, 
        num_samples = inference.ell_samples
    )

@dispatch(ProductLikelihood, Transform, FullGaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Samples from the approximate posteriors need to be transformed through the prior and then the 
        ELL is approximated using monte-carlo
    """

    likelihood_arr = likelihood.likelihood_arr

    num_likelihoods = len(likelihood_arr)
    N = Y.shape[0]

    return mv_block_monte_carlo(
        compute_ell_for_sample, 
        q_f_mu_arr, 
        q_f_var_arr, 
        fn_args = [X, Y, prior, likelihood, approximate_posterior],
        generator = inference.generator, 
        num_samples = inference.ell_samples
    )

@dispatch(BlockDiagonalGaussian, Transform, FullGaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Samples from the approximate posteriors need to be transformed through the prior and then the 
        ELL is approximated using monte-carlo
    """
    N, P = Y.shape

    variance = likelihood.variance

    # q_f_var and variance blocks should be the same dimension
    chex.assert_shape(variance, q_f_var_arr.shape)
    chex.assert_shape(Y, q_f_mu_arr.shape)

    X = X[..., None]
    Y = Y[..., None]
    q_f_mu_arr = q_f_mu_arr[..., None]

    # ELL is the sum of the individual blocks
    ell_blocks = jax.vmap(
        full_gaussian_expected_log_likelihood,
        [0, 0, 0, 0, 0],
        0
    )(X, Y, variance, q_f_mu_arr, q_f_var_arr)

    chex.assert_shape(ell_blocks, [N])

    return np.sum(ell_blocks)

