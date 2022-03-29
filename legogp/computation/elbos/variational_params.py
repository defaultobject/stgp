"""
Collects the required parts of q(u) for a given ELBO.


By convention in the single output / diagonal settings the output will be:
    mu: M x 1
    var: M x 1

Let B = the block size, N_b the number of blocks then In the multi-output/block diagonal setting: 
    mu: N_b x B x 1
    var: N_b x B x B

When required we enforce these conventations through assertions.

"""
import chex
import jax
import jax.numpy as np

from ...dispatch import dispatch, evoke
from ...utils.batch_utils import batch_over_module_types
from ..marginals import gaussian_conditional_diagional, gaussian_conditional
from ..matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec

# Import Types
from ...transforms import Transform, LinearTransform, Independent, NonLinearTransform
from ...approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, FullConjugateGaussian
from ...likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ...sparsity import FreeSparsity, Sparsity

# ================================== Dispatched q(u) ==============================

@dispatch('GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', 'NoSparsity')
def variational_params(data, approximate_posterior, likelihood, prior, sparsity):
    """
    Gaussian q(u). When the likelihood is Gaussian and no sparsity is used only the diagonal
    of q(u) is required.

    output:
        mu: Mx1
        var: Mx1
    """
    mu, var =  approximate_posterior.m, diagonal_from_cholesky(approximate_posterior.S_chol)

    chex.assert_rank(mu, 2)
    chex.assert_equal_shape([mu, var])

    return mu, var

@dispatch('DiagonalGaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', 'NoSparsity')
def variational_params(data, approximate_posterior, likelihood, prior, sparsity):
    """
    output:
        mu: Mx1
        var: Mx1
    """
    mu, var =  approximate_posterior.m, approximate_posterior.S_diag
    var = var[:, None]

    chex.assert_rank(mu, 2)
    chex.assert_equal_shape([mu, var])

    return mu, var

@dispatch('GaussianApproximatePosterior', BlockDiagonalLikelihood, 'GPPrior', 'NoSparsity')
def variational_params(data, approximate_posterior, likelihood, prior, sparsity):
    """
    With a block diagonal likelihood the approximate posterior is assumed to have the correct ordering.

    output:
        mu: N_b x B x 1
        var: N_b x B x B
    """

    raise RuntimeError()

    block_size = likelihood.block_size
    return  block_from_vec(approximate_posterior.m, block_size), block_diagonal_from_cholesky(approximate_posterior.S_chol, block_size)

@dispatch('ConjugateGaussian', DiagonalLikelihood, 'GPPrior', 'NoSparsity')
def variational_params(data, approximate_posterior, likelihood, prior, sparsity):
    """
    output:
        mu: Mx1
        var: Mx1
    """
    mu, var = approximate_posterior.surrogate.posterior(diagonal=True)
    N = mu.shape[0]

    mu = np.reshape(mu, [N, 1])
    var = np.reshape(var, [N, 1])

    return mu, var

@dispatch('ConjugateGaussian', BlockDiagonalLikelihood, 'GPPrior', Sparsity)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity):
    """
    Let B = the block size, N_b the number of blocks then
    output:
        mu: N_b x B x 1
        var: N_b x B x B
    """

    block_size = likelihood.block_size
    N = data.X.shape[0]

    mu, var = approximate_posterior.surrogate.posterior_blocks()

    # Normalize shapes
    mu = np.reshape(mu, [N, block_size, 1])
    var = np.reshape(var, [N, block_size, block_size])

    return mu, var

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior')
def variational_params(data, approximate_posterior, likelihood, prior):
    """  Single approximate posterior setting """
    sparsity = prior.sparsity
    mu, var = evoke('variational_params', approximate_posterior, likelihood, prior, sparsity)(
        data, approximate_posterior, likelihood, prior, sparsity
    ) 

    return mu, var

@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent)
def variational_params(data, approximate_posterior, likelihood, prior):
    """  Mean-field approximate posterior setting """
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = prior.num_latents

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    q_m, q_S = batch_over_module_types(
        evoke_name = 'variational_params',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [data, approx_posteriors_arr, likelihood_arr[0], latents_arr, sparsity_arr],
        fn_axes = [None, 0, None, 0, 0],
        dim = num_latents,
        out_dim  = 2
    )

    return q_m, q_S

@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent)
def variational_params(data, approximate_posterior, likelihood, prior):
    """  Full-posterior approximate posterior setting """
    return approximate_posterior.m, approximate_posterior.S_blocks

@dispatch(FullConjugateGaussian, Likelihood, Independent)
def variational_params(data, approximate_posterior, likelihood, prior):
    """  conjugate Full-posterior approximate posterior setting """
    return approximate_posterior.surrogate.posterior_blocks()
