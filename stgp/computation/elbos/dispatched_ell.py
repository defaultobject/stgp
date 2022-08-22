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
from ...approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, ApproximatePosterior
from ...core.model_types import get_model_type, LinearModel, NonLinearModel, get_non_linear_model_part

from batchjax import batch_or_loop, BatchType
from numpy.polynomial.hermite import hermgauss

def get_block_type(lik_block_size, q_block_size):
    if (lik_block_size == 1) and (q_block_size == 1):
        return 'Diagonal'
    
    return 'Blocked'

# ====================== SCALAR ELL COMPONENTS ===================
@dispatch('scalar', Gaussian, GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ Gaussian expected log likelihood component. """
    return scalar_gaussian_expected_log_likelihood(X, Y, likelihood.variance, q_f_mu, q_f_var)

@dispatch('scalar', Likelihood, GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ Expected log likelihood component approximated through quadrature. """
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

# ====================== GAUSSIAN ELLs ===================

@dispatch(BlockDiagonalGaussian, GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    # TODO: adding missing data masking
    block_size = likelihood.block_size

    #Reshape Y to match q_f_mu
    Y_blocks = block_from_vec(Y, block_size)
    X_blocks = block_from_mat(X, block_size)

    # Ensure correct shapes after vmap
    Y_blocks = Y_blocks[..., None]
    #q_f_mu = q_f_mu[..., None]

    lik_var = likelihood.variance

    ell_arr = jax.vmap(
        full_gaussian_expected_log_likelihood,
        [0, 0, 0, 0, 0],
        0
    )(X_blocks, Y_blocks, lik_var, q_f_mu, q_f_var)

    ell = np.sum(ell_arr)

    return ell


@dispatch("DiagonalGaussian", GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
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

@dispatch(DiagonalLikelihood, GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ For diagonal likelihoods adds support for missing data. """ 

    # Get nan mask for output
    mask = get_mask(Y)

    # Convert nans to zeros
    Y = mask_vector(Y, mask)

    X = X[:, None, ...]
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

@dispatch(PowerLikelihood, GaussianApproximatePosterior)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    scale_a = likelihood.a

    parent_lik = likelihood.parent

    parent_ell =  evoke('expected_log_likelihood', parent_lik, approx_posterior)(
        X, Y, q_f_mu, q_f_var, parent_lik
    )

    return parent_ell * scale_a

# ====================== ELL FOR DIFFERENT APPROXIMATE POSTERIORS ===================

# Gaussian Approximate Posterior
@dispatch(Data, Likelihood, 'GPPrior', GaussianApproximatePosterior)
def expected_log_likelihood(data, q_f_mu, q_f_var, likelihood, prior, approx_posterior, inference):
    X, Y = data.X, data.Y

    return evoke('expected_log_likelihood', likelihood, approx_posterior)(
        X, Y, q_f_mu, q_f_var, likelihood
    )

@dispatch(Likelihood, 'GPPrior', GaussianApproximatePosterior)
def expected_log_likelihood_with_xy(X, Y, q_f_mu, q_f_var, likelihood, prior, approx_posterior, inference):
    return evoke('expected_log_likelihood', likelihood, approx_posterior)(
        X, Y, q_f_mu, q_f_var, likelihood
    )

# Meanfield Approximate Posterior
@dispatch(Data, ProductLikelihood, LinearModel, MeanFieldApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    When the prior is a linear transform the approximate posterior is Gaussian and 
        q_f_mu, q_f_var will already be transformed if necessary
    """

    X, Y = data.X, data.Y

    likelihood_arr = likelihood.likelihood_arr
    approx_posteriors_arr = approximate_posterior.approx_posteriors

    # TODO: this assumes an indepedent prior
    latent_arr = prior.base_prior.parent

    Y = Y[..., None]

    ell_arr = batch_over_module_types(
        evoke_name = 'expected_log_likelihood_with_xy',
        evoke_params = [],
        module_arr = [likelihood_arr, latent_arr, approx_posteriors_arr],
        fn_params = [X, Y, q_f_mu_arr, q_f_var_arr, likelihood_arr, prior, approx_posteriors_arr, inference],
        fn_axes = [None, 1, 0, 0, 0, None, 0, None],
        dim = len(approx_posteriors_arr),
        out_dim  = 1 
    )

    chex.assert_shape(ell_arr, [len(likelihood_arr)])

    return ell_arr

# Meanfield Approximate Posterior
def compute_ell_for_sample(f, X, Y, prior, likelihood, approximate_posterior):
    """
    Args:
        f: N x P x B - sampled f
        Y: N x P - data output

    """

    chex.assert_rank(f, 3)
    chex.assert_rank(Y, 2)

    N, Q, B = f.shape
    P = Y.shape[1]

    likelihood_arr = likelihood.likelihood_arr
    num_likelihoods = len(likelihood_arr)

    chex.assert_equal(P, num_likelihoods)

    non_linear_prior_part = get_non_linear_model_part(prior)

    if non_linear_prior_part is not None:
        transformed_f = f
        for p in non_linear_prior_part:
            # Reparameterise
            # Transform through prior for each datapoint
            transformed_f = jax.vmap(
                p.forward,
                [0],
                0
            )(transformed_f)
    else:
        transformed_f = f


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

# Meanfield Gaussian with non-linear ELL Approximate Posterior
@dispatch(Data, ProductLikelihood, NonLinearModel, MeanFieldApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Samples from the approximate posteriors need to be transformed through the prior and then the 
        ELL is approximated using monte-carlo
    """

    X, Y = data.X, data.Y

    likelihood_arr = likelihood.likelihood_arr
    approx_posteriors_arr = approximate_posterior.approx_posteriors

    num_likelihoods = len(likelihood_arr)
    N, P = Y.shape
    Q = prior.base_prior.output_dim

    # Normalise shapes
    q_f_mu_arr = np.reshape(q_f_mu_arr, [Q, N])
    q_f_var_arr = np.reshape(q_f_var_arr, [Q, N])


    ell =  mv_indepentdent_monte_carlo(
        compute_ell_for_sample, 
        q_f_mu_arr, 
        q_f_var_arr, 
        fn_args = [X, Y, prior, likelihood, approximate_posterior],
        generator = inference.generator, 
        num_samples = inference.ell_samples
    )

    chex.assert_shape(ell, [N, P, 1])

    ell = np.sum(ell, axis=0)[:, 0]

    return ell

# Full Gaussian ELL Approximate Posterior
@dispatch(Data, ProductLikelihood, NonLinearModel, FullGaussianApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Samples from the approximate posteriors need to be transformed through the prior and then the 
        ELL is approximated using monte-carlo
    """

    X, Y = data.X, data.Y

    Q = prior.base_prior.output_dim
    N = Y.shape[0]

    chex.assert_rank(q_f_var_arr, 3)
    chex.assert_shape(q_f_mu_arr, [N, Q])
    chex.assert_shape(q_f_var_arr, [N, Q, Q])

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

# ===============================================================================
# ================================= ^^^ OLD ^^^ =================================
# ===============================================================================

@dispatch(Data, ProductLikelihood, Transform, ApproximatePosterior, 'Diagonal')
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    print('diagonal')
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])

    N, P, B = q_f_mu.shape

    chex.assert_equal([q_f_mu.shape[0], q_f_mu.shape[1]] , [N, P])
    chex.assert_equal([q_f_var.shape[0], q_f_var.shape[1]] , [N, P])

    model_type = get_model_type(prior)

    if False and isinstance(model_type, LinearModel):
        # check if closed form expression exists
        print('linear')
        pass
    else:
        print('non-linear')

        likelihood_arr = likelihood.likelihood_arr

        # q_f_mu is of shape:
        #   N x P x B
        # q_v_var is of shape:
        #   N x P x B x 1

        num_likelihoods = len(likelihood_arr)
        N, P = Y.shape
        Q = prior.base_prior.output_dim

        # Normalise shapes
        q_f_var = q_f_var[..., 0]

        ell =  mv_indepentdent_monte_carlo(
            compute_ell_for_sample, 
            q_f_mu, 
            q_f_var, 
            fn_args = [X, Y, prior, likelihood, approximate_posterior],
            generator = inference.generator, 
            num_samples = inference.ell_samples
        )

        chex.assert_shape(ell, [N, P, 1])

        ell = np.sum(ell, axis=0)[:, 0]

        return ell

    # use sampling

    return 0

@dispatch(Data, Likelihood, Transform, ApproximatePosterior, 'Blocked')
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    print('blocked')
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])
    chex.assert_equal([q_f_var.shape[1]], [1])

    N, Q, B = q_f_mu.shape
    P = Y.shape[1]

    model_type = get_model_type(prior)

    if isinstance(model_type, LinearModel):
        # check if closed form expression exists
        print('linear')
        pass
    else:
        print('non-linear')

        # TODO: error in q_f_var
        ell = mv_block_monte_carlo(
            compute_ell_for_sample, 
            q_f_mu, 
            q_f_var, 
            fn_args = [X, Y, prior, likelihood, approximate_posterior],
            generator = inference.generator, 
            num_samples = inference.ell_samples
        )

        chex.assert_shape(ell, [N, P, 1])

        ell = np.sum(ell, axis=0)[:, 0]

        return ell

    return 0

# ===============================================================================
# ================================= Entry Point =================================
# ===============================================================================

@dispatch(Data, Likelihood, Transform, MeanFieldApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    base_prior = prior.base_prior

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    return evoke('expected_log_likelihood', data, likelihood, model_type, approximate_posterior)(
        data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference
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


# ================================= Special Cases =================================

@dispatch(Data, GaussianProductLikelihood, LinearModel, FullGaussianApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Both q_f_mu_arr and q_f_var_arr are already in data-latent format
    We just need to mix them by W and call full_gaussian_expected_log_likelihood
    """

    chex.assert_rank([q_f_mu_arr, q_f_var_arr], [2, 3])

    X, Y = data.X, data.Y

    if False:
        breakpoint()
        if isinstance(prior, DataLatentPermutation):
            W = prior.parent.W
        else:
            W = prior.W

        # Ensure rank 2 after batching
        q_f_mu_arr = q_f_mu_arr[..., None]
        Y = Y[..., None]

        # Mix outputs by the linear transform defined in the prior
        q_f_mu_arr = jax.vmap(lambda W, f: W @ f, [None, 0])(W, q_f_mu_arr)
        q_f_var_arr = jax.vmap(lambda W, S: W @ S @ W.T, [None, 0])(W, q_f_var_arr)
    else:
        # Ensure rank 2 after batching
        q_f_mu_arr = q_f_mu_arr[..., None]

        q_f_mu_arr, q_f_var_arr = jax.vmap(lambda p, mu, var: prior.transform(mu, var), [None, 0, 0])(prior, q_f_mu_arr, q_f_var_arr)

        chex.assert_rank([q_f_mu_arr, q_f_var_arr], [3, 3])

        # Ensure rank 2 after batching
        Y = Y[..., None]
        chex.assert_rank(Y, 3)

    variance = np.diag(likelihood.variance)

    # ELL is the sum of the individual blocks
    ell_blocks = jax.vmap(
        full_gaussian_expected_log_likelihood,
        [None, 0, None, 0, 0],
        0
    )(X, Y, variance, q_f_mu_arr, q_f_var_arr)

    chex.assert_shape(ell_blocks, [q_f_mu_arr.shape[0]])

    return np.sum(ell_blocks)

@dispatch(Data, BlockDiagonalGaussian, Transform, FullGaussianApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    X, Y = data.X, data.Y

    # TODO
    # Y has shape Nt x Ns x P
    # At each timestep we need latent-data order because of how the state space is representated
    # To convert to latent-data order we just need to stack each spatials observations
    Y = np.reshape(np.transpose(Y, [0, 2, 1]), [-1, data.Ns*data.P]) 

    # Ensure rank 2 after batching
    Y = Y[..., None]

    # X should be P x N x D
    #chex.assert_rank(X, 3)

    # Y should be N x P x 1
    chex.assert_rank(Y, 3)

    N, P, _ = Y.shape

    variance = likelihood.variance

    num_blocks = likelihood.num_blocks
    block_size = likelihood.block_size

    #Y_vec = stack_rows(Y)
    #Y = block_from_vec(Y_vec, block_size)

    # X is in latent-data order. First we transpose to convert to data-latent order and then
    # stack the rows through the reshape
    #X = np.reshape(np.transpose(X, [1, 0, 2]), [num_blocks, -1, X.shape[-1]])

    # Ensure correct dimensions after batching
    q_f_mu_arr = q_f_mu_arr[..., None]

    # ELL is the sum of the individual blocks
    ell_blocks = jax.vmap(
        full_gaussian_expected_log_likelihood,
        [None, 0, 0, 0, 0],
        0
    )(X, Y, variance, q_f_mu_arr, q_f_var_arr)

    chex.assert_shape(ell_blocks, [num_blocks])

    return np.sum(ell_blocks)


@dispatch(TransformedData, Likelihood, Transform, ApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    base_data = data.base_data

    base_ell =  evoke('expected_log_likelihood', base_data, likelihood, prior, approximate_posterior)(
        data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference
    )

    log_jac = data.log_jacobian(data.Y_base)

    # Ignores nans
    log_jac = np.nan_to_num(log_jac, 0.0)
    log_jac = np.sum(log_jac)

    return base_ell + log_jac

