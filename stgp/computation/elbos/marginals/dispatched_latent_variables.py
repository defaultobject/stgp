import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full, gaussian_conditional_blocks, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve, batched_block_diagional
from ...permutations import left_permute_mat, data_order_to_output_order, permute_vec, permute_mat, unpermute_vec, unpermute_mat
from ....core import Block, get_block_dim

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation, DataLatentPermutation
from ....transforms.latent_variable import LatentVariable, UncertainInput
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateApproximatePosterior, FullConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior

from .linear_marginals import linear_marginal_blocks

# ==== LV part ====

def latent_variable_marginal(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool, predict: bool):
    # GP prior for which we are adding LV to
    gp_prior = prior.parent.parent[0]
    lv_prior = prior.parent.parent[1]

    gp_q = approximate_posterior.approx_posteriors[0]
    lv_q = approximate_posterior.approx_posteriors[1]

    # only support single GPs atm
    assert type(gp_prior).__name__ == 'GPPrior'
    assert type(lv_prior).__name__ == 'GPPrior'

    # collect data and sparisty

    X = data.X
    Z_gp = gp_prior.sparsity.Z
    Z_lv = lv_prior.sparsity.Z

    # predict the latent variable
    gp_mu = q_m[0]
    gp_S_chol = q_S_chol[0][0]

    lv_mu = q_m[1]
    lv_S_chol = q_S_chol[1][0]
    lv_S = lv_S_chol @ lv_S_chol.T

    if predict:
        lv_pred_mu, lv_pred_var = evoke('marginal_prediction_blocks', lv_q, likelihood, lv_prior, lv_prior.sparsity, whiten=whiten)(
            XS, lv_prior.sparsity, lv_mu[..., None], lv_S_chol[None, None, ...], lv_q, likelihood, lv_prior, lv_prior.sparsity, None, whiten
        )
        lv_pred_mu = lv_pred_mu[None, ..., 0]
        lv_pred_var = lv_pred_var[None, ..., 0, 0]

    else:
        lv_pred_mu, lv_pred_var = lv_mu, lv_S
        lv_pred_mu = lv_pred_mu[None, ...]
        lv_pred_var = np.diag(lv_pred_var)[None, :, None]



    Kzz = gp_prior.covar(Z_gp, Z_gp)

    K_x1x2 = np.zeros([1, lv_pred_mu.shape[1], Z_gp.shape[0]])
    K_x2 = np.zeros([1, Z_gp.shape[0], 1])

    print('lv_pred_mu: ', lv_pred_mu.shape)
    print('lv_pred_var: ', lv_pred_var.shape)

    print('K_x1x2: ',K_x1x2.shape)
    print('K_x2: ', K_x2.shape)
    print('prior.deep_kernel.lengthscales: ', prior.deep_kernel.lengthscales)


    # deep kernel is only defined on the second input kernel of gp_prior.kernel
    # x1, X2, pm_x1, pm_x2, pk_x1, pk_x2, pk_x1x2
    Kxz_2 = prior.deep_kernel._K_with_pm(
        XS, Z_gp[:, 1:], lv_pred_mu, Z_gp[:, 1:][None, ...], lv_pred_var, K_x1x2, K_x2
    )
    Kxz_1 = gp_prior.kernel.k1.K(XS[:, :1], Z_gp[:, :1])

    #element wise multiplication for the product kernel
    Kxz = np.multiply(Kxz_1, Kxz_2)

    Kxx_var = prior.deep_kernel.K_diag(lv_pred_var)  
    print('Kxx_var: ', Kxx_var)



    #print(prior.deep_kernel._K_with_pm( XS, Z_gp[:, 1:], lv_pred_mu, lv_pred_mu, np.zeros_like(lv_pred_var), K_x1x2, K_x2))
    #breakpoint()

    mu, var = gaussian_conditional_diagional(
        XS, 
        Z_gp, 
        Kzz, 
        Kxz, 
        Kxx_var, 
        gp_mu,
        gp_S_chol,
        np.zeros(Z_gp.shape[0])[:, None],
        np.zeros(X.shape[0])[:, None]
    )

    var = var - mu**2
    print('Kxz: ', Kxz)
    print('var: ', var)

    mu = mu[:, None, ]
    var = var[ :, None,  None, ...]

    chex.assert_rank([mu, var], [3, 4])

    return mu, var


@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    .
    """
    # use GP predictive equations for approximate_psoterior[0] but use the MMK kernel 

    if whiten:
        raise NotImplementedError()

    return latent_variable_marginal(
        data.X, data,  q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten, False
    )

@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=False)
def marginal_prediction(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples=None, posterior=False):

    # use GP predictive equations for approximate_psoterior[0] but use the MMK kernel 

    if whiten:
        raise NotImplementedError()

    q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, whiten)(
        data, approximate_posterior, likelihood, prior.base_prior, whiten
    )


    return latent_variable_marginal(
        XS, data,  q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten, True
    )



