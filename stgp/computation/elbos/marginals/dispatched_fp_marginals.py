"""
Full posteriors are transformed in the expected log likelihood / prediction functions so these marginals are only dealing with the base prior (where the GPs are defined).

There are two types of prior that we handle:

    1) when the base GPs are independent 
    2) when the base GPs are joint

And the following types of expected log likelihood/transform

    a) Diagonal 
    b) (tbd) Blocked (for example Aggregated Data/Transforms)
"""
import chex
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from .... import settings
from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_blocks
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, stack_rows, cholesky_solve, cholesky, add_jitter
from ..prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ, prior_mean_X, prior_covar_X
from ...permutations import data_order_to_output_order
from ...integrals.approximators import mv_block_monte_carlo

from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Joint
from ....approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, FullConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ....transforms import DataLatentPermutation, MultiOutput

def _get_wrapped_base_prior(prior):
    base_prior = prior.base_prior
    if isinstance(base_prior, Joint):
        base_prior = JointDataLatentPermutation(base_prior)
    elif isinstance(base_prior, Independent):
        base_prior = IndependentDataLatentPermutation(base_prior)
    else:
        raise RuntimeError()

    return base_prior

@dispatch('DataLatentBlockDiagonalApproximatePosterior', Likelihood, Transform)
def marginal(data, approximate_posterior, likelihood, prior):
    """ 
    approximate_posterior is already is data latent format so does not need to updated. 
    """
    return approximate_posterior.m, approximate_posterior.S_blocks

@dispatch('prediction', FullGaussianApproximatePosterior, Likelihood, Transform, Sparsity, False)
@dispatch('prediction', FullGaussianApproximatePosterior, Likelihood, Transform, Sparsity, True)
def marginal(XS, data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, whiten):
    """ 
    approximate_posterior is already is in latent data format and so needs to be converted to data-latent format. 
    """

    if whiten:
        base_prior = prior.base_prior
        Z = base_prior.get_Z()
        chex.assert_rank(Z, 3)

        Kzz = base_prior.b_covar(Z, Z)
        chex.assert_rank(Kzz, 2)

        Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))
        chex.assert_equal_shape([Kzz_chol, q_S])

        q_m, q_S =  Kzz_chol @ q_m, Kzz_chol @ q_S


    base_prior = prior.base_prior

    Q = base_prior.output_dim
    M = sparsity[0].shape[0]
    D = XS.shape[-1]
    NS = XS.shape[0]

    # Variational parameters are in latent-data format
    chex.assert_shape(q_m, [M * Q, 1])
    chex.assert_shape(q_S, [M * Q, M * Q])

    # Get all Z in latent-data format
    Z_all = base_prior.get_Z()
    chex.assert_shape(Z_all, [Q, M, D])

    # Convert XS to latent_data format
    XS_tiled = np.tile(XS, [Q, 1, 1])

    # Z does not need to be ordered, only X
    # Compute non permuted full covariance - this will be block diagonal
    K_zz = prior.np_b_covar(Z_all, Z_all)
    chex.assert_shape(K_zz, [Q*M, Q*M])

    # Compute Kxz with x permutated into data-latent format
    # Left permute x, and do not permute Z
    Kxz_p = prior.lp_rb_covar(XS, Z_all)
    chex.assert_shape(Kxz_p, [Q*NS, Q*M])

    # Compute the block diagonals of the permutated Kxx
    # TODO: stop tiling XS here
    K_xx_p = prior.b_full_var_blocks(
        XS_tiled,
        1,
        Q
    )

    chex.assert_shape(K_xx_p, [NS, Q, Q])

    mean_Z = prior.b_mean(Z_all)
    mean_XS = prior.mean(XS)

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

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, whiten):

    assert isinstance(prior, DataLatentPermutation)

    m = q_m
    S = q_S_chol @ q_S_chol.T

    Ns = m.shape[0]

    # Use the base prior as the transformation happens in thre ELL for Full posteriors
    num_latents = prior.base_prior.output_dim

    # X is shaped so that all outputs are grouped together
    # We need to instead group by each input

    m_p = prior.permute_vec(m, num_latents)
    S_p = prior.permute_mat(S, num_latents)

    m_p = np.reshape(m_p, [-1, num_latents])

    # Extract block diagonals
    S_blocks = get_block_diagonal(S_p, num_latents)

    # Assert shapes are correct
    chex.assert_shape(m_p, [Ns/num_latents, num_latents])
    chex.assert_shape(S_blocks, [Ns/num_latents, num_latents, num_latents])

    return m_p, S_blocks

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', True)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, whiten):
    # first reparameterise and then we can treat as use

    # reparameterise
    base_prior = prior.base_prior

    Z = base_prior.get_Z()
    chex.assert_rank(Z, 3)

    Kzz = base_prior.b_covar(Z, Z)
    chex.assert_rank(Kzz, 2)

    Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    chex.assert_equal_shape([Kzz_chol, q_S_chol])

    q_m, q_S_chol =  Kzz_chol @ q_m, Kzz_chol @ q_S_chol

    # we have reparemeterised the approximate posterior so we can now treat it as unwhitened
    fn = evoke('marginal', approximate_posterior, likelihood, prior, sparsity[0], False)

    return fn(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, False
    ) 


@dispatch(FullConjugateGaussian, Likelihood, Transform, 'NoSparsity', False)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, whiten):
    """ CVI already returns q_m, q_S in blocks """
    return q_m, q_S


@dispatch('FullGaussianApproximatePosterior', Likelihood, Transform, FreeSparsity, False)
def marginal(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity[0])

    mu, var =  fn(
        data.X, data, approximate_posterior, likelihood, prior, sparsity, whiten
    )

    return mu, var

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, False)
def marginal(data, q_m, q_S, approximate_posterior, likelihood, prior, whiten):

    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.base_prior.get_sparsity_list()

    base_prior = _get_wrapped_base_prior(prior)

    fn = evoke('marginal', approximate_posterior, likelihood, base_prior, sparsity_arr[0], whiten)

    return fn(
        data, q_m, q_S, approximate_posterior, likelihood, base_prior, sparsity_arr, whiten
    ) 

@dispatch('latents', FullGaussianApproximatePosterior, Likelihood, Transform, False)
@dispatch('latents', FullGaussianApproximatePosterior, Likelihood, Transform, True)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten):

    if diagonal is False:
        raise NotImplementedError()

    sparsity_arr = prior.base_prior.get_sparsity_list()

    prior = _get_wrapped_base_prior(prior)

    fn = evoke('marginal', 'prediction', approximate_posterior, likelihood, prior, sparsity_arr[0], whiten)

    if isinstance(approximate_posterior, FullConjugateGaussian):
        # Fullconjugate Gaussian does not need the variational params to predict
        latent_mu, latent_var =  fn(
            XS, data, approximate_posterior, likelihood, prior, sparsity_arr, whiten
        ) 
    else:
        q_m, q_S = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, whiten)(
            data, approximate_posterior, likelihood, prior, whiten
        )

        latent_mu, latent_var =  fn(
            XS, data, q_m, q_S , approximate_posterior, likelihood, prior, sparsity_arr, whiten
        ) 

    return latent_mu, latent_var

@dispatch('samples', FullGaussianApproximatePosterior, Likelihood, Transform, False)
@dispatch('samples', FullGaussianApproximatePosterior, Likelihood, Transform, True)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):
    if diagonal is False:
        raise NotImplementedError()

    sparsity_arr = prior.base_prior.get_sparsity_list()

    base_prior = _get_wrapped_base_prior(prior)

    latent_mu, latent_var = evoke('marginal', 'latents', approximate_posterior, likelihood, base_prior, whiten)(
        XS, data, approximate_posterior, likelihood, base_prior, inference, whiten, diagonal
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

    return mu

@dispatch('prediction', FullGaussianApproximatePosterior, Likelihood, Transform, False)
@dispatch('prediction', FullGaussianApproximatePosterior, Likelihood, Transform, True)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal):

    mu = evoke('marginal', 'samples', approximate_posterior, likelihood, prior, whiten)(
        XS, data, approximate_posterior, likelihood, prior, inference, whiten, diagonal
    )

    second_moment =  mu**2

    mu = np.mean(mu, axis=0)
    second_moment = np.mean(second_moment, axis=0)

    var = second_moment - np.square(mu)

    # Fix shapes
    mu = (mu.T)[..., None]
    var = (var.T)[..., None]

    return mu, var

