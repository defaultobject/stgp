"""
When computing the ELBO we typically implement the marginal q(f) in two stages:
    1) collect q(u)
    2) compute q(f) = \int p(f|u) q(u) du

This is because it makes it easier to compute gradients of q(u) through the ELL term, 
    and makes it easier to plug in different conjugate models for q(u).

However when predicting it is usually more efficient to use the surrogate model specific prediction functions.
"""
import chex
import jax
import jax.numpy as np
import chex

from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform
from ....approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, FullConjugateGaussian, ConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood, BlockDiagonalGaussian
from ....sparsity import FreeSparsity, Sparsity, NoSparsity, SpatialSparsity


@dispatch('prediction', ConjugateGaussian, DiagonalLikelihood, 'GPPrior', Sparsity)
def marginal(XS, data, approximate_posterior, likelihood, prior, sparsity):
    N = XS.shape[0]

    mu, var = approximate_posterior.surrogate.predict_f(XS, diagonal=True)

    mu = np.reshape(mu, [N, 1])
    var = np.reshape(var, [N, 1])

    return mu, var

@dispatch('prediction', FullConjugateGaussian, Likelihood, Transform, Sparsity)
def marginal(XS, data, approximate_posterior, likelihood, prior, sparsity):
    # Returns mu, var in data-latent ordering
    mu, var =  approximate_posterior.surrogate.predict_blocks(
        XS, 1, prior.num_latents, diagonal=False
    )

    return mu, var


