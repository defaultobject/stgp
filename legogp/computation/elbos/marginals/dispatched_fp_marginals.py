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
from ....approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, FullConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ....transforms import DataLatentPermutation

@dispatch('DataLatentBlockDiagonalApproximatePosterior', Likelihood, Transform)
def marginal(data, approximate_posterior, likelihood, prior):
    """ tbd. """
    return approximate_posterior.m, approximate_posterior.S_blocks

@dispatch('prediction', FullGaussianApproximatePosterior, Likelihood, Transform, Sparsity)
def marginal(XS, data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity):
    Q = prior.num_latents
    M = sparsity[0].shape[0]
    D = XS.shape[-1]
    NS = XS.shape[0]

    # Variational parameters are in latent-data format
    chex.assert_shape(q_m, [M * Q, 1])
    chex.assert_shape(q_S, [M * Q, M * Q])

    # Get all Z in latent-data format
    Z_all = prior.get_Z()
    chex.assert_shape(Z_all, [Q, M, D])

    # Convert XS to latent_data format
    XS_tiled = np.tile(XS, [Q, 1, 1])

    # Z does not need to be ordered, only X
    # Compute non permuted full covariance - this will be block diagonal
    K_zz = prior.np_full_covar(Z_all, Z_all)
    chex.assert_shape(K_zz, [Q*M, Q*M])

    # Compute Kxz with x permutated into data-latent format
    # Left permute x, and do not permute Z
    Kxz_p = prior.lp_ls_full_covar(XS, Z_all)
    chex.assert_shape(Kxz_p, [Q*NS, Q*M])

    # Compute the block diagonals of the permutated Kxx
    # TODO: stop tiling XS here
    K_xx_p = prior.blocks_var(
        XS_tiled,
        1,
        Q
    )
    chex.assert_shape(K_xx_p, [NS, Q, Q])

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
        Z_all, 
        K_zz, 
        Kxz_p, 
        K_xx_p, 
        q_m,
        q_S,
        mean_Z,
        mean_XS,
    )

    return _m, _S

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity')
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity):
    m = q_m
    S = q_S_chol @ q_S_chol.T

    Ns = m.shape[0]
    num_latents = prior.num_latents

    # X is shaped so that all outputs are grouped together
    # We need to instead group by each input

    m_p = prior.permute_vec(m)
    S_p = prior.permute_mat(S)

    m_p = np.reshape(m_p, [-1, num_latents])

    # Extract block diagonals
    S_blocks = get_block_diagonal(S_p, num_latents)

    # Assert shapes are correct
    chex.assert_shape(m_p, [Ns/num_latents, num_latents])
    chex.assert_shape(S_blocks, [Ns/num_latents, num_latents, num_latents])

    return m_p, S_blocks

@dispatch(FullConjugateGaussian, Likelihood, Transform, 'NoSparsity')
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity):
    return q_m, q_S


@dispatch('FullGaussianApproximatePosterior', Likelihood, Transform, FreeSparsity)
def marginal(data, approximate_posterior, likelihood, prior, sparsity):
    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity[0])

    mu, var =  fn(
        data.X, data, approximate_posterior, likelihood, prior, sparsity
    )

    return mu, var

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior):

    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.get_sparsity_list()

    fn = evoke('marginal', approximate_posterior, likelihood, prior, sparsity_arr[0])

    return fn(
        data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity_arr
    ) 

@dispatch('latents', FullGaussianApproximatePosterior, Likelihood, Transform)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal):

    if diagonal is False:
        raise NotImplementedError()

    sparsity_arr = prior.get_sparsity_list()


    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity_arr[0])

    if isinstance(approximate_posterior, FullConjugateGaussian):
        # Fullconjugate Gaussian does not need the variational params to predict
        latent_mu, latent_var =  fn(
            XS, data, approximate_posterior, likelihood, prior, sparsity_arr
        ) 
    else:
        q_m, q_S = evoke('variational_params', approximate_posterior, likelihood, prior.latent_obj)(
            data, approximate_posterior, likelihood, prior
        )

        latent_mu, latent_var =  fn(
            XS, data, q_m, q_S , approximate_posterior, likelihood, prior, sparsity_arr
        ) 

    return latent_mu, latent_var


@dispatch('prediction', FullGaussianApproximatePosterior, Likelihood, Transform)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal):

    if diagonal is False:
        raise NotImplementedError()

    sparsity_arr = prior.get_sparsity_list()


    latent_mu, latent_var = evoke('marginal', 'latents', approximate_posterior, likelihood, prior)(
        XS, data, approximate_posterior, likelihood, prior, inference, diagonal
    )

    # Compute transformed q(f)
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

