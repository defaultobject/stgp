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

from .linear_marginals import linear_marginal_blocks

@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    """ Computes the diagonal of q(f) = ∫ p(f | u) q(u) du """
    chex.assert_rank([m, S_chol], [2, 2])
    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

    return gaussian_conditional_diagional(
        XS, 
        data.X, 
        prior.covar(sparsity.Z, sparsity.Z), 
        prior.covar(XS, sparsity.Z), 
        prior.var(XS), 
        m,
        S_chol,
        prior.mean(sparsity.Z),
        prior.mean(XS),
    )


@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=True)
def marginal_prediction_blocks(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    """ Computes the diagonal of q(f) = ∫ p(f | u) q(u) du """
    chex.assert_rank([m, S_chol], [2, 2])
    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

    #TODO: only works with zero mean gps
    return whitened_gaussian_conditional_diagional(
        XS, 
        data.X, 
        prior.covar(sparsity.Z, sparsity.Z), 
        prior.covar(XS, sparsity.Z), 
        prior.var(XS)[:, 0], 
        m,
        S_chol
    )



@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = sparsity
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = XS.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]
    whiten_arr = [whiten for q in range(num_latents)]
    out_block_arr = [out_block_dim for q in range(num_latents)]

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal_prediction_blocks',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [XS, data, q_m, q_S_chol, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
        fn_axes = [None, None, 0, 0, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2,
        evoke_kwargs = {'whiten': whiten}
    )

    # TODO: this should be fixed in the lower level predictoins but hacked here for now

    marginal_var = marginal_var[..., None]


    marginal_mu = np.transpose(marginal_mu, [1, 0, 2])
    marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])

    chex.assert_shape(marginal_mu, [N, prior.output_dim,  out_block_dim])
    chex.assert_shape(marginal_var, [N, 1, prior.output_dim*out_block_dim, prior.output_dim*out_block_dim])


    return marginal_mu, marginal_var

@dispatch(ApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):

    return linear_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim, whiten, XS=XS, sparsity=sparsity
    )


@dispatch(ApproximatePosterior, Likelihood, Transform, Sparsity, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, Transform, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, linear_model_part, sparsity[0], whiten=whiten)(
        XS, data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, sparsity, out_block_dim, whiten
    ) 


    if isinstance(model_type, LinearModel):
        return val
    
    breakpoint()
    raise NotImplementedError()

# ========================= Predictions =========================

@dispatch('latents', MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, out_block_dim, whiten):

    sparsity_list = prior.base_prior.get_sparsity_list()
    latents = prior.base_prior
    q_m, q_S_chol = approximate_posterior.get_variational_params()

    mu, var = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, latents, sparsity_list[0], whiten=whiten)(
        XS, data, q_m, q_S_chol, approximate_posterior, likelihood, latents, sparsity_list, out_block_dim , whiten
    )

    return mu, var

# ================================== Marginal Covars ==============================

@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=False)
def marginal_prediction_covar(X1, X2, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):

    K12 = prior.kernel.K(X1, X2)
    K1z = prior.kernel.K(X1, sparsity.Z)
    Kz2 = prior.kernel.K(sparsity.Z, X2)
    Kzz = prior.kernel.K(sparsity.Z, sparsity.Z)

    return gaussian_conditional_covar(
        X1, X2, sparsity.Z,
        Kzz, 
        K1z,
        Kz2,
        K12,
        m,
        S_chol
    )


@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, Sparsity, whiten=False)
def marginal_prediction_covar(XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = sparsity
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N1 = XS_1.shape[0]
    N2 = XS_2.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]
    whiten_arr = [whiten for q in range(num_latents)]
    out_block_arr = [out_block_dim for q in range(num_latents)]

    # Compute q(f) for each output
    marginal_var = batch_over_module_types(
        evoke_name = 'marginal_prediction_covar',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [XS_1, XS_2, data, q_m, q_S_chol, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
        fn_axes = [None, None, None, 0, 0, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 1,
        evoke_kwargs = {'whiten': whiten}
    )

    # TODO: this should be fixed in the lower level predictoins but hacked here for now

    marginal_var = marginal_var[..., None]

    marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])

    #chex.assert_shape(marginal_var, [N, 1, prior.output_dim*out_block_dim, prior.output_dim*out_block_dim])


    return marginal_var

@dispatch(ApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=False)
def marginal_prediction_covar(XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):

    # TODO: assuming indenity transforms

    var_parent  = evoke('marginal_prediction_covar', approximate_posterior, likelihood, prior.base_prior, sparsity[0], whiten=whiten)(
        XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior.base_prior, sparsity, out_block_dim, whiten
    ) 

    return var_parent
