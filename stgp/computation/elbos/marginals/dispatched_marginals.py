import chex
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from ...core import GPPrior
from ...transforms import Transform
from ...dispatch import dispatch, evoke
from ...transforms import LinearTransform, Independent, NonLinearTransform
from ...utils.batch_utils import batch_over_module_types
from ..marginals import gaussian_conditional_diagional, gaussian_conditional
from .prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ, prior_mean_X, prior_covar_X
from ..matrix_ops import diagonal_from_cholesky, get_block_diagonal
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ...approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior
from ...sparsity import FreeSparsity, Sparsity
from ..permutations import data_order_to_output_order

@dispatch('GaussianApproximatePosterior', 'GPPrior', 'NoSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    return approximate_posterior.m, diagonal_from_cholesky(approximate_posterior.S_chol)

@dispatch('GaussianApproximatePosterior', 'GPPrior', 'FullSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    # TODO: this is a hack
    fn = evoke('marginal', 'prediction', approximate_posterior, prior, sparsity)

    return fn(
        X, X, approximate_posterior, prior, sparsity
    )

@dispatch('prediction', 'GaussianApproximatePosterior', 'GPPrior', FreeSparsity)
def marginal(XS, X, approximate_posterior, prior, sparsity):
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    return gaussian_conditional_diagional(
        XS, 
        X, 
        prior.covar(sparsity.Z, sparsity.Z)[0], 
        prior.covar(XS, sparsity.Z)[0], 
        prior.var(XS)[0], 
        m,
        S_chol,
        prior.mean(sparsity.Z)[0],
        prior.mean(XS)[0],
    )

@dispatch('full_prediction', 'GaussianApproximatePosterior', 'GPPrior', FreeSparsity)
def marginal(XS, X, approximate_posterior, prior, sparsity):
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    return gaussian_conditional(
        XS, 
        X, 
        prior.covar(sparsity.Z, sparsity.Z)[0], 
        prior.covar(XS, sparsity.Z)[0], 
        prior.covar(XS, XS)[0], 
        m,
        S_chol,
        prior.mean(sparsity.Z)[0],
        prior.mean(XS)[0],
    )

@dispatch('ConjugateGaussian', 'GPPrior', 'NoSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    mu, var = approximate_posterior.surrogate.predict_f(X, diagonal=True)
    return mu[..., None], var[..., None]

@dispatch('prediction', 'ConjugateGaussian', 'GPPrior', 'NoSparsity')
def marginal(XS, X, approximate_posterior, prior, sparsity):
    mu, var = approximate_posterior.surrogate.predict_f(XS, diagonal=True)
    return mu[..., None], var[..., None]

@dispatch('FullConjugateGaussian', Transform, Sparsity)
def marginal(X, approximate_posterior, prior, sparsity):
    breakpoint()
    return approximate_posterior.surrogate.predict_blocks(X, diagonal=True)

@dispatch('prediction', 'FullConjugateGaussian', Transform, Sparsity)
def marginal(XS, X, approximate_posterior, prior, sparsity):
    return approximate_posterior.surrogate.predict_blocks(XS, diagonal=True)


@dispatch('FullGaussianApproximatePosterior', Transform, FreeSparsity)
def marginal(X, approximate_posterior, prior, sparsity):
    fn = evoke('marginal', 'prediction', approximate_posterior, prior, sparsity[0])

    mu, var =  fn(
        X, X, approximate_posterior, prior, sparsity
    )

    return mu, var

@dispatch('FullGaussianApproximatePosterior', Transform, 'NoSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    m = approximate_posterior.m
    S = approximate_posterior.S

    num_latents = prior.num_latents
    num_outputs = prior.output_dim

    Ns = m.shape[0]
    N = X.shape[0]

    # X is shaped so that all outputs are grouped together
    # We need to instead group by each input

    # Create permutation matrix
    P = data_order_to_output_order(num_latents, N)

    # Rearrange m and S
    m_p = P @ m
    S_p = P @ S @ P.T

    m_p = np.reshape(m_p, [-1, num_latents])

    # Extract block diagonals
    S_blocks = get_block_diagonal(S_p, num_latents)

    # Assert shapes are correct
    chex.assert_shape(m_p, [Ns/num_latents, num_latents])
    chex.assert_shape(S_blocks, [Ns/num_latents, num_latents, num_latents])

    return m_p, S_blocks


@dispatch('prediction', 'FullGaussianApproximatePosterior', Transform, Sparsity)
def marginal(XS, X, approximate_posterior, prior, sparsity):
    # Compute Kzz, Kxz, Kxs_diag, mean_x, mean_xs
    m = approximate_posterior.m
    S_chol = approximate_posterior.S_chol

    fix_shape = lambda v: np.reshape(v, [v.shape[0]*v.shape[1], 1])

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



@dispatch(FullGaussianApproximatePosterior, Transform)
def marginal(X, approximate_posterior, prior):
    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.get_sparsity_list()

    fn = evoke('marginal', approximate_posterior, prior, sparsity_arr[0])

    return fn(
        X, approximate_posterior, prior, sparsity_arr
    ) 


@dispatch('prediction', FullGaussianApproximatePosterior, Transform)
def marginal(XS, X, approximate_posterior, prior, inference):
    latents = prior.latent_obj
    sparsity_arr = latents.get_sparsity_list()

    # TODO: figure out how to handle sparsity here
    fn = evoke('marginal', 'prediction', approximate_posterior, latents, sparsity_arr[0])

    latent_mu, latent_var =  fn(
        XS, X, approximate_posterior, latents, sparsity_arr
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



