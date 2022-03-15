import chex
import jax
import jax.numpy as np

from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform
from ....approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood, BlockDiagonalGaussian
from ....sparsity import FreeSparsity, Sparsity

@dispatch('ConjugateGaussian', DiagonalLikelihood, 'GPPrior', Sparsity)
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    mu, var = approximate_posterior.surrogate.predict_f(X, diagonal=True)
    return mu[..., None], var[..., None]

@dispatch('prediction', 'ConjugateGaussian', DiagonalLikelihood, 'GPPrior', Sparsity)
def marginal(XS, X, approximate_posterior, likelihood, prior, sparsity):
    mu, var = approximate_posterior.surrogate.predict_f(XS, diagonal=True)

    return mu[..., None], var[..., None]

@dispatch('ConjugateGaussian', BlockDiagonalLikelihood, 'GPPrior', Sparsity)
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    block_size = likelihood.block_size
    mu, var = approximate_posterior.surrogate.predict_blocks(X, block_size, diagonal=False)
    return mu[..., None], var

@dispatch('FullConjugateGaussian', Likelihood, Transform, Sparsity)
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    Q = prior.num_latents
    mu, var = approximate_posterior.surrogate.predict_blocks(
        X, 1, Q, diagonal=False
    )
    return mu, var

@dispatch('FullConjugateGaussian', BlockDiagonalGaussian, Transform, Sparsity)
def marginal(X, approximate_posterior, likelihood, prior, sparsity):
    Q = prior.num_latents
    num_blocks = likelihood.num_blocks
    block_size = likelihood.block_size

    # TODO: this is assuming the same number of inducing points per latent function
    M = sparsity[0].Z.shape[0]

    mu, var = approximate_posterior.surrogate.predict_blocks(
        X, M, block_size, diagonal=False
    )
    return mu, var

@dispatch('prediction', 'FullConjugateGaussian', Likelihood, Transform, Sparsity)
def marginal(XS, X, approximate_posterior, likelihood, prior, sparsity):
    # Returns mu, var in data-latent ordering
    mu, var =  approximate_posterior.surrogate.predict_blocks(
        XS, 1, prior.num_latents, diagonal=False
    )

    return mu, var


