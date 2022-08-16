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

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', True)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'FullSparsity', False)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'NoSparsity', True)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """ For computational reasons we return S_chol """
    mu, var_chol =  approximate_posterior.m, approximate_posterior.S_chol

    chex.assert_rank([mu, var_chol], [2, 2])

    return mu, var_chol

#@dispatch('GaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
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

@dispatch('DiagonalGaussianApproximatePosterior', DiagonalLikelihood, 'GPPrior', 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
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

@dispatch('GaussianApproximatePosterior', BlockDiagonalLikelihood, 'GPPrior', 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """
    With a block diagonal likelihood the approximate posterior is assumed to have the correct ordering.

    output:
        mu: N_b x B x 1
        var: N_b x B x B
    """

    raise RuntimeError()

    block_size = likelihood.block_size
    return  block_from_vec(approximate_posterior.m, block_size), block_diagonal_from_cholesky(approximate_posterior.S_chol, block_size)

@dispatch('ConjugateGaussian', DiagonalLikelihood, 'GPPrior', 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
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

@dispatch('ConjugateGaussian', DiagonalLikelihood, 'GPPrior', 'SpatialSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """
    output:
        mu: NtxNs
        var: NtxNsxNs
    """
    mu, var = approximate_posterior.surrogate.posterior_blocks()
    Nt = mu.shape[0]
    Ns = mu.shape[1]

    mu = np.reshape(mu, [Nt, Ns])
    var = np.reshape(var, [Nt, Ns, Ns])

    return mu, var

@dispatch('ConjugateGaussian', BlockDiagonalLikelihood, 'GPPrior', Sparsity, False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """
    Let B = the block size, N_b the number of blocks then
    output:
        mu: N_b x B x 1
        var: N_b x B x B
    """

    block_size = likelihood.block_size

    mu, var = approximate_posterior.surrogate.posterior_blocks()

    # Normalize shapes
    mu = np.reshape(mu, [-1, block_size, 1])
    var = np.reshape(var, [-1, block_size, block_size])

    return mu, var

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', False)
@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', True)
def variational_params(data, approximate_posterior, likelihood, prior, whiten):
    """  Single approximate posterior setting """
    sparsity = prior.sparsity

    mu, var = evoke('variational_params', approximate_posterior, likelihood, prior, sparsity, whiten)(
        data, approximate_posterior, likelihood, prior, sparsity, whiten
    ) 

    return mu, var

#================== MEAN FIELD ==========================
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, False)
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, True)
def variational_params(data, approximate_posterior, likelihood, prior, whiten):
    """  Mean-field approximate posterior setting. Collect parameters across all components q(u_q) """
    base_prior = prior.base_prior
    num_latents = base_prior.output_dim

    latents_arr = base_prior.parent
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = base_prior.get_sparsity_list()

    likelihood_arr = likelihood.likelihood_arr

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]

    q_m, q_S = batch_over_module_types(
        evoke_name = 'variational_params',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_params = [data, approx_posteriors_arr, likelihood_arr[0], latents_arr, sparsity_arr, whiten_arr],
        fn_axes = [None, 0, None, 0, 0, 0],
        dim = num_latents,
        out_dim  = 2
    )

    return q_m, q_S

#================== DENSE FULL POSTERIOR ==========================
@dispatch(FullConjugateGaussian, Likelihood, Transform, 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """  conjugate Full-posterior approximate posterior setting """
    return approximate_posterior.surrogate.posterior_blocks()


@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, True)
def variational_params(data, approximate_posterior, likelihood, prior, whiten):
    """  Full-posterior approximate posterior setting """

    sparsity_arr = prior.base_prior.get_sparsity_list()

    #TODO: assuming same sparsity across all latents
    sparsity = sparsity_arr[0]

    mu, var = evoke('variational_params', approximate_posterior, likelihood, prior, sparsity, whiten)(
        data, approximate_posterior, likelihood, prior, sparsity, whiten
    ) 

    return mu, var
