import chex
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec
from ..prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ, prior_mean_X, prior_covar_X
from ...permutations import data_order_to_output_order
from ...integrals.approximators import mv_block_monte_carlo

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform
from ....approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity



@dispatch('prediction', 'FullGaussianApproximatePosterior', Likelihood, Transform, Sparsity)
def marginal(XS, X, approximate_posterior, likelihood, prior, sparsity):

    # Compute Kzz, Kxz, Kxs_diag, mean_x, mean_xs
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    fix_shape = lambda v: np.reshape(v, [v.shape[0]*v.shape[1], 1])

    # Compute q(F) = \int p(F | U) q(U) dU
    m, S =  gaussian_conditional(
        XS, 
        X, 
        block_diag(*prior_covar_ZZ(prior)), 
        block_diag(*prior_covar_XZ(prior, XS)), 
        block_diag(*prior_covar_X(prior, XS)), 
        m,
        S_chol,
        fix_shape(prior_mean_Z(prior)),
        fix_shape(prior_mean_X(prior, XS)),
    )

    # Create permutation matrix
    num_latents = prior.num_latents
    NS = XS.shape[0]
    N = m.shape[0]
    P = data_order_to_output_order(num_latents, XS.shape[0])

    # Rearrange m and S
    m_p = P @ m
    S_p = P @ S @ P.T

    m_p = np.reshape(m_p, [-1, num_latents])

    # Extract block diagonals
    S_blocks = get_block_diagonal(S_p, num_latents)

    # Assert shapes are correct
    chex.assert_shape(m_p, [N/num_latents, num_latents])
    chex.assert_shape(S_blocks, [N/num_latents, num_latents, num_latents])
    

    return m_p, S_blocks



@dispatch('FullGaussianApproximatePosterior', Likelihood, Transform, FreeSparsity)
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity[0])

    mu, var =  fn(
        X, X, approximate_posterior, likelihood, prior, sparsity
    )

    return mu, var


@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform)
def marginal(X, approximate_posterior, likelihood, prior):

    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.get_sparsity_list()
    
    fn = evoke('marginal', approximate_posterior, likelihood, prior, sparsity_arr[0])

    return fn(
        X, approximate_posterior, likelihood, prior, sparsity_arr
    ) 


@dispatch('prediction', FullGaussianApproximatePosterior, Likelihood, Transform)
def marginal(XS, X, approximate_posterior, likelihood, prior, inference, diagonal):

    if diagonal is False:
        raise NotImplementedError()

    latents = prior.latent_obj
    sparsity_arr = latents.get_sparsity_list()

    # TODO: generalize sparsity?
    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, latents, sparsity_arr[0])

    latent_mu, latent_var =  fn(
        XS, X, approximate_posterior, likelihood, latents, sparsity_arr
    ) 

    vmaped_prior_forard =  jax.vmap(prior.forward, [1], 0)

    mu = mv_block_monte_carlo(
        lambda f, fn: fn(f),
        latent_mu,
        latent_var,
        fn_args=[vmaped_prior_forard],
        generator = inference.generator, 
        num_samples = inference.prediction_samples,
        average=False
    )

    second_moment =  mu**2

    mu = np.mean(mu, axis=0)
    second_moment = np.mean(second_moment, axis=0)

    var = second_moment - np.square(mu)

    # Fix shapes
    mu = (mu.T)[..., None]
    var = (var.T)[..., None]

    return mu, var

