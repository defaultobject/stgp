import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part

def linear_marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, out_block_dim: int, whiten: bool, XS=None, sparsity=None):
    """ Recursively compute the transformed linear marginal. """

    parent_prior = prior.parent

    if prior.in_block_dim is None:
        # we do not need to worry about the block size
        parent_out_block_dim = out_block_dim
    else:
        parent_out_block_dim = max(out_block_dim, prior.in_block_dim)

    # get parent transformed value

    if XS is None:
        mu_parent, var_parent  = evoke('marginal_blocks', approximate_posterior, likelihood, parent_prior, whiten=whiten)(
            data, q_m, q_S, approximate_posterior, likelihood, parent_prior, parent_out_block_dim, whiten
        ) 
    else:
        mu_parent, var_parent  = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, parent_prior, sparsity[0], whiten=whiten)(
            XS, data, q_m, q_S, approximate_posterior, likelihood, parent_prior, sparsity, parent_out_block_dim, whiten
        ) 

    if prior.out_block_dim != None:
        breakpoint()

    # and transform 
    if var_parent.shape[-1] == 1 and out_block_dim == 1:
        mu_p, var_p = prior.transform_diagonal(mu_parent, var_parent)
    else:
        breakpoint()
        raise NotImplementedError()

    return mu_p, var_p
