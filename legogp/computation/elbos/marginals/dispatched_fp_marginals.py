import chex
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_blocks
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, stack_rows, cholesky_solve
from ..prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ, prior_mean_X, prior_covar_X
from ...permutations import data_order_to_output_order
from ...integrals.approximators import mv_block_monte_carlo

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform
from ....approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ....transforms import DataLatentPermutation


@dispatch('prediction', 'FullGaussianApproximatePosterior', Likelihood, Transform, Sparsity)
def marginal(XS, X, approximate_posterior, likelihood, prior, sparsity):
    Q = prior.num_latents

    # Get variational parameters in latent-data format
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    # Get all Z in latent-data format
    Z_all = prior.get_Z()

    # Convert XS to latent_data format
    XS_tiled = np.tile(XS, [Q, 1, 1])

    # Z does not need to be ordered, only X
    K_zz = prior.np_full_covar(Z_all, Z_all)

    # Compute Kxz with x permutated into data-latent format
    Kxz_p = prior.lp_ls_full_covar(XS, Z_all)

    # Compute the block diagonals of the permutated Kxx
    # TODO: stop tiling XS here
    K_xx_p = prior.blocks_var(
        XS_tiled,
        1,
        Q
    )

    mean_Z = prior.mean(Z_all)
    mean_XS = prior.s_mean(XS)

    # Compute q(F) = \int p(F | U) q(U) dU
    # Comput blocks of
    #val = K_xx - Kxz_p @ cholesky_solve(K_chol, Kxz_p.T)
    # TODO: assuming mean is zero
    _m, _S =  gaussian_conditional_blocks(
        1, 
        Q, 
        XS, 
        X, 
        K_zz, 
        Kxz_p, 
        K_xx_p, 
        m,
        S_chol,
        mean_Z,
        mean_XS,
    )

    return _m, _S

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

    sparsity_arr = prior.get_sparsity_list()

    # TODO: generalize sparsity?
    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity_arr[0])

    latent_mu, latent_var =  fn(
        XS, X, approximate_posterior, likelihood, prior, sparsity_arr
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

