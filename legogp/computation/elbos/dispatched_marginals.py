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
from ...approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior

@dispatch('GaussianApproximatePosterior', 'GPPrior', 'NoSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    return approximate_posterior.m, diagonal_from_cholesky(approximate_posterior.S_chol)

@dispatch('ConjugateGaussian', 'GPPrior', 'NoSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    mu, var = approximate_posterior.surrogate.predict_f(X, diagonal=True)
    return mu[..., None], var[..., None]

@dispatch('prediction', 'ConjugateGaussian', 'GPPrior', 'NoSparsity')
def marginal(XS, X, approximate_posterior, prior, sparsity):
    mu, var = approximate_posterior.surrogate.predict_f(XS, diagonal=True)
    return mu[..., None], var[..., None]

@dispatch('prediction', 'GaussianApproximatePosterior', 'GPPrior', 'NoSparsity')
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

@dispatch('FullGaussianApproximatePosterior', Transform, 'NoSparsity')
def marginal(X, approximate_posterior, prior, sparsity):
    m = approximate_posterior.m
    S = approximate_posterior.S

    num_latents = prior.num_latents
    num_outputs = prior.output_dim

    N = m.shape[0]

    # X is shaped so that all outputs are grouped together
    # We need to instead group by each input

    # Create permutation matrix
    # Create permutation matrix
    num_latents = prior.num_latents
    NS = X.shape[0]
    N = m.shape[0]

    i = np.hstack([np.arange(i,N, NS) for i in range(NS)])
    P = np.eye(N)[i]

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


@dispatch('prediction', 'FullGaussianApproximatePosterior', Transform, 'NoSparsity')
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

    i = np.hstack([np.arange(i,N, NS) for i in range(NS)])
    P = np.eye(N)[i]

    #P = np.eye(N)
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

@dispatch(MeanFieldApproximatePosterior, Independent)
def marginal(X, approximate_posterior, prior):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()

    num_latents = len(sparsity_arr)
    N = X.shape[0]

    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_params = [X, approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_axes = [None, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var


@dispatch('prediction', MeanFieldApproximatePosterior, Independent)
def marginal(XS, X, approximate_posterior, prior, inference):
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()

    num_latents = len(sparsity_arr)
    N = XS.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal',
        evoke_params = ['prediction'],
        module_arr = [approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_params = [XS, X, approx_posteriors_arr, latents_arr, sparsity_arr],
        fn_axes = [None, None, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2
    )

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch(MeanFieldApproximatePosterior, LinearTransform)
def marginal(X, approximate_posterior, prior):

    latents = prior.latents

    num_latents = len(latents.latents)
    N = X.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', approximate_posterior, latents)(
        X, approximate_posterior, latents
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch('FullGaussianApproximatePosterior', Transform)
def marginal(X, approximate_posterior, prior):
    # TODO: figure out how to handle sparsity here
    fn = evoke('marginal', approximate_posterior, prior, 'NoSparsity')

    return fn(
        X, approximate_posterior, prior, None
    ) 


@dispatch('prediction', 'FullGaussianApproximatePosterior', Transform)
def marginal(XS, X, approximate_posterior, prior, inference):

    latents = prior.latents
    sparsity_arr = latents.get_sparsity_list()

    # TODO: figure out how to handle sparsity here
    fn = evoke('marginal', 'prediction', approximate_posterior, latents, 'NoSparsity')

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

    #mu = np.transpose(mu, [1, 0, 2])
    #second_moment = np.transpose(second_moment, [1, 0, 2])


    var = second_moment - np.square(mu)

    return mu, var



@dispatch(MeanFieldApproximatePosterior, NonLinearTransform)
def marginal(X, approximate_posterior, prior):
    latents = prior.latents

    return   evoke('marginal', approximate_posterior, latents)(
        X, approximate_posterior, latents
    ) 

@dispatch('prediction', MeanFieldApproximatePosterior, LinearTransform)
def marginal(XS, X, approximate_posterior, prior, inference):

    latents = prior.latents

    num_latents = len(latents.latents)
    N = XS.shape[0]

    # Compute q(f) for each output
    marginal_mu, marginal_var = evoke('marginal', 'prediction', approximate_posterior, latents)(
        XS, X, approximate_posterior, latents
    )

    # Mix outputs by the linear transform defined in the prior
    marginal_mu, marginal_var = prior.transform_diagonal(marginal_mu, marginal_var)

    chex.assert_shape(marginal_mu, [num_latents, N, 1])
    chex.assert_shape(marginal_var, [num_latents, N, 1])

    return marginal_mu, marginal_var

@dispatch('prediction', MeanFieldApproximatePosterior, NonLinearTransform)
def marginal(XS, X, approximate_posterior, prior, inference):
    latents = prior.latents

    latent_mu, latent_var =   evoke('marginal', 'prediction', approximate_posterior, latents)(
        XS, X, approximate_posterior, latents
    ) 

    vmaped_prior_forard =  jax.vmap(prior.forward, [1], 0)

    mu = mv_indepentdent_monte_carlo(
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

    mu = np.transpose(mu, [1, 0, 2])
    second_moment = np.transpose(second_moment, [1, 0, 2])


    var = second_moment - np.square(mu)
    
    return mu, var
