import chex
import jax
import jax.numpy as np
import objax

from ...dispatch import dispatch, evoke
from ..matrix_ops import block_from_vec, block_from_mat, stack_rows

from ...data import Data, TransformedData
from ...transforms import LinearTransform, Independent, NonLinearTransform, Transform, DataLatentPermutation, MultiOutput
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import get_mask, mask_vector, mask_matrix, get_same_shape_mask
from ...utils.utils import get_batch_type
from ...likelihood import ProductLikelihood, DiagonalLikelihood, Likelihood, DiagonalGaussian, Gaussian, BlockDiagonalGaussian, GaussianProductLikelihood, PowerLikelihood
from .expected_log_likelihoods import scalar_gaussian_expected_log_likelihood, gaussian_expected_log_likelihood, full_gaussian_expected_log_likelihood
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ..integrals.samples import approximate_expectation
from ...approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, ApproximatePosterior
from ...core.model_types import get_model_type, LinearModel, NonLinearModel, get_non_linear_model_part, get_block_type

from batchjax import batch_or_loop, BatchType
from numpy.polynomial.hermite import hermgauss



# ====================== SCALAR ELL COMPONENTS ===================
@dispatch(Gaussian)
def scalar_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ Gaussian expected log likelihood component. """
    return scalar_gaussian_expected_log_likelihood(X, Y, likelihood.variance, q_f_mu, q_f_var)

# TODO: rename this
@dispatch(GaussianProductLikelihood, 'Blocked')
def element_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    lik_var = np.diag(likelihood.variance)
    return full_gaussian_expected_log_likelihood(X, Y, lik_var, q_f_mu, q_f_var)


# ====================== GAUSSIAN ELLs ===================

@dispatch(BlockDiagonalGaussian, 'Blocked')
@dispatch(BlockDiagonalGaussian, 'Diagonal')
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    N = Y.shape[0]

    # TODO: adding missing data masking
    block_size = likelihood.block_size

    #Reshape Y to match q_f_mu
    Y_blocks = block_from_vec(Y, block_size)
    X_blocks = block_from_mat(X, block_size)

    # Ensure correct shapes after vmap
    Y_blocks = Y_blocks[..., None]
    q_f_mu = np.reshape(q_f_mu, [N, block_size, block_size])

    lik_var = likelihood.variance

    ell_arr = jax.vmap(
        full_gaussian_expected_log_likelihood,
        [0, 0, 0, 0, 0],
        0
    )(X_blocks, Y_blocks, lik_var, q_f_mu, q_f_var)

    ell = np.sum(ell_arr)

    return ell


#@dispatch("DiagonalGaussian", GaussianApproximatePosterior)
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ Special case for diagonal Gaussian"""
    N = X.shape[0]

    # Get nan mask for output
    mask = get_mask(Y)

    # Convert nans to zeros
    Y = mask_vector(Y, mask)

    X = X[:, None, ...]
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

@dispatch(DiagonalLikelihood, 'Diagonal')
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    """ For diagonal likelihoods adds support for missing data. """ 
    chex.assert_rank([Y, q_f_mu, q_f_var], [2, 2, 3])

    # q_f_var is diagional so we fix the shapes so all shapes batch
    q_f_var = q_f_var[..., 0]
    chex.assert_equal_shape([Y, q_f_mu, q_f_var])

    # Get nan mask for output
    mask = get_mask(Y)

    # Convert nans to zeros
    Y = mask_vector(Y, mask)

    # Ensure rank 2 after batching
    X = X[:, None, ...]
    Y = Y[..., None]
    q_f_mu = q_f_mu[..., None]
    q_f_var = q_f_var[..., None]

    fn = evoke('scalar_expected_log_likelihood', likelihood)

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

@dispatch(PowerLikelihood, 'Blocked')
@dispatch(PowerLikelihood, 'Diagonal')
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    scale_a = likelihood.a

    parent_lik = likelihood.parent

    parent_ell =  evoke('expected_log_likelihood', parent_lik, block_type)(
        X, Y, q_f_mu, q_f_var, parent_lik, block_type
    )

    return parent_ell * scale_a


@dispatch(ProductLikelihood, 'Blocked')
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    chex.assert_rank([Y, q_f_mu, q_f_var], [2, 3, 4])

    # As this ELL does not decompose across latents and datapoints
    #   the nan-handling must be handled lower down

    # Ensure rank 2 after batching
    X = X[:, None, ...]
    Y = Y[..., None]
    q_f_var = q_f_var[:, 0, ...]

    fn = evoke('element_expected_log_likelihood', likelihood, block_type)

    # Compute ELL for each datapoint
    ell_arr = jax.vmap(
        fn,
        [0, 0, 0, 0, None],
        0
    )(X, Y, q_f_mu, q_f_var, likelihood)


    # Only sums the ELL terms without missing data
    ell = np.sum(ell_arr)

    return ell



# ====================== ELL FOR DIFFERENT APPROXIMATE POSTERIORS ===================

def compute_ell_for_sample(transformed_f, X, Y, prior, likelihood, approximate_posterior):
    """
    Args:
        transformed_f: N x P x B - sampled  and transformed f
        Y: N x P - data output

    """

    chex.assert_rank(transformed_f, 3)
    chex.assert_rank(Y, 2)

    N, Q, B = transformed_f.shape
    P = Y.shape[1]

    likelihood_arr = likelihood.likelihood_arr
    num_likelihoods = len(likelihood_arr)

    chex.assert_equal(P, num_likelihoods)

    # Y and F must be rank 2 when they are passed to log_likelihood
    # When vmapping one dimension is lost so extent here
    Y = Y[..., None]
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

    #return np.sum(ll_arr)
    return ll_arr

@dispatch(Data, Likelihood, 'GPPrior', ApproximatePosterior, 'Diagonal')
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])
    chex.assert_equal([q_f_mu.shape[1], q_f_var.shape[1]], [1, 1])

    if False: 
        # use monte carlo / quadrature
        pass
    else:
        # single output already 
        ell = evoke('single_output_expected_log_likelihood', likelihood, block_type)(
            X, Y, q_f_mu[:, 0, ...], q_f_var[:, 0, ...], likelihood, block_type
        )

        return ell

    breakpoint()


@dispatch(Data, ProductLikelihood, Transform, ApproximatePosterior, 'Diagonal')
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])

    N, P, B = q_f_mu.shape

    chex.assert_equal([q_f_mu.shape[0], q_f_mu.shape[1]] , [N, P])
    chex.assert_equal([q_f_var.shape[0], q_f_var.shape[1]] , [N, P])

    model_type = get_model_type(prior)


    # TODO: there is a choice here between quadrature and monte-carlo estimation
    # TODO: need to check if a likelihood has a closed form ELL
    if isinstance(model_type, LinearModel):
        # check if closed form expression exists
        # batch over each output
        likelihood_arr = likelihood.likelihood_arr

        block_arr = [block_type for l in likelihood_arr]

        # Ensure rank 2 after batching
        Y = Y[..., None]
        ell_arr = batch_over_module_types(
            evoke_name = 'single_output_expected_log_likelihood',
            evoke_params = [],
            module_arr = [likelihood_arr, block_arr],
            fn_params = [X, Y, q_f_mu, q_f_var, likelihood_arr, block_arr],
            fn_axes = [None, 1, 1, 1, 0, 0],
            dim = P,
            out_dim  = 1 
        )
        chex.assert_shape(ell_arr, [P])

        ell = np.sum(ell_arr)

        return ell

    else:
        likelihood_arr = likelihood.likelihood_arr

        # q_f_mu is of shape:
        #   N x P x B
        # q_v_var is of shape:
        #   N x P x B x 1

        num_likelihoods = len(likelihood_arr)
        N, P = Y.shape
        Q = prior.base_prior.output_dim

        ell = approximate_expectation(
            compute_ell_for_sample, 
            q_f_mu, 
            q_f_var, 
            prior = prior,
            fn_args = [X, Y, prior, likelihood, approximate_posterior],
            generator = inference.generator, 
            num_samples = inference.ell_samples,
            block_type = block_type,
            average = True
        )

        chex.assert_shape(ell, [N, P, 1])

        ell = np.sum(ell)

        return ell

    raise RuntimeError()

@dispatch(Data, Likelihood, Transform, ApproximatePosterior, 'Blocked')
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    print('blocked')
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])
    chex.assert_equal([q_f_var.shape[1]], [1])

    N, Q, B = q_f_mu.shape
    P = Y.shape[1]

    model_type = get_model_type(prior)

    if  isinstance(model_type, LinearModel):
        # check if closed form expression exists
        ell = evoke('single_output_expected_log_likelihood', likelihood, block_type)(
           X, Y, q_f_mu, q_f_var, likelihood, block_type
        )

        ell = np.sum(ell)

        return ell
    else:

        ell = approximate_expectation(
            compute_ell_for_sample, 
            q_f_mu, 
            q_f_var, 
            prior = prior,
            fn_args = [X, Y, prior, likelihood, approximate_posterior],
            generator = inference.generator, 
            num_samples = inference.ell_samples,
            block_type = block_type,
            average = True
        )

        chex.assert_shape(ell, [N, P, 1])

        ell = np.sum(ell)

        return ell

    raise RuntimeError()


# ===============================================================================
# ================================= Entry Point =================================
# ===============================================================================

@dispatch(Data, Likelihood, 'GPPrior', ApproximatePosterior)
@dispatch(Data, Likelihood, Transform, ApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    chex.assert_rank([q_f_mu_arr, q_f_var_arr], [3, 4])
    base_prior = prior.base_prior

    # get correct ELL corresponding to the blocks
    q_block_size = q_f_var_arr.shape[-1]
    lik_block_size = likelihood.block_size

    block_type_p = get_block_type(lik_block_size, q_block_size)

    return evoke('expected_log_likelihood', data, likelihood, prior, approximate_posterior, block_type_p)(
        data.X, data.Y, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference, block_type_p
    )

@dispatch(Data, ProductLikelihood, MultiOutput, MeanFieldApproximatePosterior)
@dispatch(Data, ProductLikelihood, MultiOutput, FullGaussianApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Multioutput assumes that
        Data Is 
    """
    ell_arr = []

    # TODO: what about data?
    for p in range(prior.output_dim):
        prior_p = prior.parent[p]
        likelihood_p = likelihood.likelihood_arr[p]
        q_f_mu_p = q_f_mu_arr[p]
        q_f_var_p = q_f_var_arr[p]

        chex.assert_rank([q_f_mu_p, q_f_var_p], [3, 4])

        q_block_size = q_f_var_p.shape[-1]
        lik_block_size = likelihood.block_size

        assert lik_block_size <= q_block_size

        X_p = data.X
        Y_p = data.Y[:, p][:, None]

        block_type_p = get_block_type(lik_block_size, q_block_size)

        ell_p =  evoke('expected_log_likelihood', data, likelihood_p, prior_p, approximate_posterior, block_type_p)(
            X_p, Y_p, q_f_mu_p, q_f_var_p, likelihood_p, prior_p, approximate_posterior, inference, block_type_p
        )
        ell_arr.append(ell_p)

    return np.sum(np.array(ell_arr))

@dispatch(TransformedData, Likelihood, Transform, ApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    base_data = data.base_data

    base_ell =  evoke('expected_log_likelihood', base_data, likelihood, prior, approximate_posterior)(
        data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference
    )

    # convert nans to zero
    # these will have a jacobian of zero and so will not contribute to the sum
    log_jac = data.log_jacobian(np.nan_to_num(data.Y_base, nan=1.0))

    # ensure it does not constribute
    nan_mask = get_same_shape_mask(data.Y_base)

    #ignore nans
    log_jac = nan_mask * log_jac

    log_jac = np.sum(log_jac)

    return base_ell + log_jac
