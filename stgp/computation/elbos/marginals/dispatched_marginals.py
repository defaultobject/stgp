"""
Dispatched functions for computing:
    1) q(u)
    2) q(f) = \int p(f | u) q(u) du

To make computing natural gradients easier we compute the ELL is computed by:
    1) Collecting appropriate paramters from the approximate posterior q(u):
        - (i.e the diagonal, block diagonal, full covariance, etc)
    2) Passing these to the appropiate marginal to compute q(f)
    3) Compute the ELL

This file contains the dispatched method for computing both q(u) and q(f)

The way marginals are computed is very general to support 
    - blocked likelihoods,
    - meanfield and full gaussian approximate posteriors,
    - linear / non linear transformations
    - differential operator transformations

With these a wide variety of variational GP based models can be constructed.
"""
import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full, gaussian_conditional_blocks, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve
from ...permutations import left_permute_mat

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation, DataLatentPermutation
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateApproximatePosterior
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior

from .linear_marginals import linear_marginal_blocks
# ================================== Dispatched q(f) ==============================
@dispatch(ConjugateApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    N = q_m.shape[0]

    # ensure correct shape
    q_m = np.reshape(q_m, [N, 1, out_block_dim])
    q_S = np.reshape(q_S, [N, 1, out_block_dim, out_block_dim])

    return q_m, q_S

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    """ Catch all for single latent functions with no sparsity and no whitening"""
    N = q_m.shape[0]

    if out_block_dim == 1:
        q_S = diagonal_from_cholesky(q_S_chol)
    elif q_S_chol.shape[-1] < out_block_dim:
        raise RuntimeError()
    elif q_S_chol.shape[-1] > out_block_dim:
        # TODO: subsample
        raise RuntimeError()

    # ensure correct shape
    
    q_m = q_m[..., None]
    q_S = np.reshape(q_S, [N, 1, out_block_dim, out_block_dim])

    return q_m, q_S

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    """ Catch all for single latent functions with no sparsity but with whitening"""
    N = q_m.shape[0]
    chex.assert_rank(q_S_chol, 2)
    chex.assert_shape(q_S_chol, [q_m.shape[0], q_m.shape[0]])

    Kzz = prior.covar(sparsity.Z, sparsity.Z)
    Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    if out_block_dim == 1:
        q_m = Kzz_chol @ q_m
        q_S = diagonal_from_cholesky(Kzz_chol @ q_S_chol)
    else:
        raise NotImplementedError()

    # ensure correct shape
    q_m = q_m[..., None]
    q_S = np.reshape(q_S, [N, 1, out_block_dim, out_block_dim])

    return q_m, q_S

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):

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

    m_p = m_p[..., None]
    S_blocks = S_blocks[:, None, ...]

    # Assert shapes are correct
    chex.assert_shape(m_p, [Ns/num_latents, num_latents, 1])
    chex.assert_shape(S_blocks, [Ns/num_latents, 1, num_latents, num_latents])

    return m_p, S_blocks

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
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
    fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity[0], whiten=False)

    return fn(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, False
    ) 


@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', Sparsity, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten):
    """ Catch all for single latent functions with no sparsity"""
    M = q_m.shape[0]

    # TODO: block dim

    mu, var =  evoke(
        'marginal_prediction_blocks', approximate_posterior, likelihood, 'GPPrior', Sparsity, whiten=False 
    )(
        data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_dim, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var
# ================================== Dispatched q(f) ==============================

@dispatch(FullGaussianApproximatePosterior, ProductLikelihood, Independent, whiten=True)
@dispatch(FullGaussianApproximatePosterior, ProductLikelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim, whiten):

    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.base_prior.get_sparsity_list()

    base_prior = get_permutated_prior(prior)


    fn = evoke('marginal_blocks', approximate_posterior, likelihood, base_prior, sparsity_arr[0], whiten=whiten)

    mu, var = fn(
        data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block_dim, whiten
    ) 
    chex.assert_rank([mu, var], [3, 4])

    return mu, var



@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=True)
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim, whiten):
    latents_arr = prior.parent
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = prior.get_sparsity_list()
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N = data.X.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]
    out_block_arr = [out_block_dim for q in range(num_latents)]

    # TODO: pre-compute Kzz and Kzx so that any kernel can be used in the latents and batching can still be used.
    # Compute q(f) for each output
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'marginal_blocks',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [data, q_m, q_S_chol, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
        fn_axes = [None, 0, 0, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 2,
        evoke_kwargs = {'whiten': whiten}
    )

    chex.assert_rank([marginal_mu, marginal_var], [4, 5])
    # fix shapes
    # each component will return rank (3, 4). But each component is only one ouput so we can remove that axis
    #   and reshape into the proper shape
    marginal_mu = marginal_mu[..., 0]
    marginal_var = marginal_var[..., 0]

    marginal_mu = np.transpose(marginal_mu, [1, 0, 2])
    marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])

    chex.assert_shape(marginal_mu, [N, num_latents,  out_block_dim])
    chex.assert_shape(marginal_var, [N, num_latents, out_block_dim, out_block_dim])

    return marginal_mu, marginal_var

@dispatch(ApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim: int, whiten: bool):
    if prior.is_base:
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        base_prior = get_permutated_prior(prior)

        fn = evoke('marginal_blocks', approximate_posterior, likelihood, base_prior, sparsity_type, whiten=whiten)

        mu, var = fn(data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block_dim, whiten)
        chex.assert_rank([mu, var], [3, 4])
        return mu, var

    else:
        assert out_block_dim == 1

        # get D(t) from the kalman filter here
        # and compute D(s) manually -- since K is separable

        # need a linear approximation of the kalman filter 
        #to approximate dL/dm wrt with dL/dY^tilde dY^tilde/dm

        # TODO: move into dispatched_marginal_predictors
        # TODO: check and fix permutations

        if True:

            mu, var = approximate_posterior.approx_posteriors[0].surrogate.posterior(diagonal=False, full_state=True)

            # only pick the first Q dimensions

            var = var[:, None, ...]


            XS = data.X
            N = XS.shape[0]
            chex.assert_shape(mu, [N, prior.output_dim,  out_block_dim])
            chex.assert_shape(var, [N, 1, prior.output_dim*out_block_dim, prior.output_dim*out_block_dim])

            return mu, var

            sparsity_arr = prior.base_prior.get_sparsity_list()
            sparsity_type = sparsity_arr[0]

            Z = sparsity_arr[0].Z
            Q = prior.derivative_kernel.output_dim

            XS = data.X
            NS = XS.shape[0]

            var_fn = prior.base_prior.covar

            # in data-latent format
            K_xx_diag = jax.vmap(lambda x: prior.derivative_kernel.K_from_fn(x[None, ...], x[None, ...], var_fn))(XS)

            # in data-latent format
            K_xz = jax.vmap(lambda x, z: prior.derivative_kernel.K_from_fn(x[None, ...], z[None, ...], var_fn), [0, 0])(XS, Z)

            # in data-latent format
            K_zz = jax.vmap(lambda x: var_fn(x[None, ...], x[None, ...]))(Z)

            #mu, var = gaussian_conditional(data.X, Z, K_zz, K_xz, np.zeros([NS * Q, NS * Q]), q_m[0], q_S_chol[0], np.zeros_like(q_m[0]), np.zeros([NS * Q, 1]))

            # compute marginal q(f) \int p(f | u) q(u) df 
            fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior.base_prior, whiten=whiten)

            q_m, q_S = fn(
                data, q_m, q_S_chol, approximate_posterior, likelihood, prior.base_prior, out_block_dim, whiten
            ) 
            chex.assert_rank([q_m, q_S], [3, 4])
           
            if whiten:
                raise NotImplementedError()
            else:
                #breakpoint()
                #sqrt as gaussian_conditional requires the cholesky of q_S
                kzz = K_zz[0]
                kxz = K_xz[0]
                print(kxz @ np.eye(kxz.shape[1], kzz.shape[0]) @ q_m[0])
                breakpoint()

                mu, var = jax.vmap(
                    lambda x, z, kzz, kxz, kxx, qm, qs, muz, mux:
                        gaussian_conditional(x[None, ...], z[None, ...], kzz, kxz @ np.eye(kxz.shape[1], kzz.shape[0]), kxx, qm, np.sqrt(qs), muz, mux),
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]
                )(data.X, Z, K_zz, K_xz, K_xx_diag, q_m, q_S[..., 0], np.zeros_like(q_m), np.zeros([NS, Q, 1]))

                var = var[:, None, ...]
                breakpoint()

            N = XS.shape[0]
            chex.assert_shape(mu, [N, prior.output_dim,  out_block_dim])
            chex.assert_shape(var, [N, 1, prior.output_dim*out_block_dim, prior.output_dim*out_block_dim])

            return mu, var

        if False:
            sparsity_arr = prior.base_prior.get_sparsity_list()
            sparsity_type = sparsity_arr[0]

            Z = sparsity_arr[0].Z
            Q = prior.derivative_kernel.output_dim

            XS = data.X
            NS = XS.shape[0]

            var_fn = prior.base_prior.covar

            # compute marginal q(f) \int p(f | u) q(u) df 
            fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior.base_prior, whiten=whiten)

            q_m, q_S = fn(
                data, q_m, q_S_chol, approximate_posterior, likelihood, prior.base_prior, out_block_dim, whiten
            ) 
            chex.assert_rank([q_m, q_S], [3, 4])

            if whiten:
                raise NotImplementedError()
            else:

                def mean_fn(XX):
                    return jax.vmap(
                        lambda x, z, qm, qs:
                            gaussian_conditional(
                                x,
                                z,
                                prior.base_prior.covar(z, z), 
                                prior.base_prior.covar(x, z), 
                                prior.base_prior.covar(x, x), 
                                qm, 
                                np.sqrt(qs),
                                np.zeros_like(z), 
                                np.zeros_like(x)
                            ),
                        [0, 0, 0, 0]
                        )(XX[:, None, ...], Z[:, None, ...], q_m, q_S[..., 0])[0][..., 0]


                print(mean_fn(XS))
                print(jax.jacrev(mean_fn)(XS))
                breakpoint()
                mu = prior.derivative_mean.mean_blocks_from_fn(XS, mean_fn)
                breakpoint()

                var_fn = lambda : jax.vmap(
                    lambda x, z, kzz, kxz, kxx, qm, qs, muz, mux:
                        gaussian_conditional(x[None, ...], z[None, ...], kzz, kxz @ np.eye(kxz.shape[1], kzz.shape[0]), kxx, qm, np.sqrt(qs), muz, mux),
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]
                )(data.X, Z, K_zz, K_xz, K_xx_diag, q_m, q_S[..., 0], np.zeros_like(q_m), np.zeros([NS, Q, 1]))[0]

                breakpoint()

                var = var[:, None, ...]

            N = XS.shape[0]
            chex.assert_shape(mu, [N, prior.output_dim,  out_block_dim])
            chex.assert_shape(var, [N, 1, prior.output_dim*out_block_dim, prior.output_dim*out_block_dim])

            return mu, var
            
        if False:
            sparsity_arr = prior.base_prior.get_sparsity_list()
            sparsity_type = sparsity_arr[0]

            Z = sparsity_arr[0].Z
            Q = prior.derivative_kernel.output_dim

            XS = data.X
            NS = XS.shape[0]

            var_fn = prior.base_prior.covar

            # in data-latent format
            K_xx_diag = jax.vmap(lambda x: prior.derivative_kernel.K_from_fn(x[None, ...], x[None, ...], var_fn))(XS)

            # latent-data forma
            K_xz = prior.derivative_kernel.K_from_fn(XS, Z, var_fn)

            K_xz = left_permute_mat(K_xz, Q)

            # latent-data format but Z does not need to be permuted
            K_zz = var_fn(Z, Z)

            H = np.eye(K_xz.shape[1], K_zz.shape[0])
            K_xz = K_xz @ H

            # tile Z
            Z_tiled = np.tile(Z[None, ...], [Q, 1, 1])

            if True:

                if True:
                    if whiten:
                        mu, var = whitened_gaussian_conditional_full(data.X, Z, K_zz, K_xz, np.zeros([NS * Q, NS * Q]), q_m[0], q_S_chol[0])
                    else:
                        breakpoint()
                        mu, var = gaussian_conditional(data.X, Z, K_zz, K_xz, np.zeros([NS * Q, NS * Q]), q_m[0], q_S_chol[0], np.zeros_like(q_m[0]), np.zeros([NS * Q, 1]))

                    var = K_xx_diag + get_block_diagonal(var, Q)
                     
                    mu = block_from_vec(mu, 3)

                    return mu[..., None], var[:, None, ...]

            # TODO: assuming mean is zero
            mu, var =  gaussian_conditional_blocks(
                1, 
                Q, 
                XS, 
                Z_tiled, 
                K_zz, 
                K_xz , 
                K_xx_diag, 
                q_m[0],
                q_S_chol[0],
                np.zeros_like(q_m),
                np.zeros([XS.shape[0], 1]),
            )

            mu = mu[..., None]
            var = var[:, None, ...]

            N = XS.shape[0]
            chex.assert_shape(mu, [N, prior.output_dim,  out_block_dim])
            chex.assert_shape(var, [N, 1, prior.output_dim*out_block_dim, prior.output_dim*out_block_dim])

            return mu, var


@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim: int, whiten: bool):
    """ Recursively compute the transformed linear marginal. """

    return linear_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim, whiten, XS=None
    )


    
# ===============================================================================================
# ===============================================================================================
# ========================================  ENTRY POINTs ========================================
# ===============================================================================================
# ===============================================================================================


# list of linear priors
@dispatch(ApproximatePosterior, Likelihood, list, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, list, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block_dim: int, whiten: bool):
    """ 
    Prior is a list of transformed to be comptued Recursively. Simple loop through and collect the results.

    Note we do not use batching and this method will only really be used when the transforms in the list are 
        different, and hence batching wont be applicable anyway.

    Recursively compute the transformed linear marginal.  
    """
    mu_list, var_list = [], []
    for p in prior:
        mu_p, var_p  = evoke('marginal_blocks', approximate_posterior, likelihood, p, whiten=whiten)(
            data, q_m, q_S_chol, approximate_posterior, likelihood, p, out_block_dim, whiten
        ) 

        mu_list.append(mu_p)
        var_list.append(var_p)

    return mu_list, var_list

# ============================ SINGLE OUTPUT APPROXIMATE POSTERIOR ENTRY POINT ============================

@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=True)
@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    tbd.
    """

    out_block_size = likelihood.block_size

    sparsity = prior.sparsity

    # we are only processing the linear part so we can assume that it is linear
    mu, var = evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_size, whiten
    ) 

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

# ============================ MEANFIELD APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # get output block size. NonLinear transforms are applied elementwise so only need the likelihood
    #   block size
    out_block_size = likelihood.block_size

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block_size, whiten
    ) 

    return val[0], val[1]


# ============================ FULL GAUSSIAN APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    out_block_size = max(
        likelihood.block_size,
        prior.base_prior.output_dim
    )

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block_size, whiten
    ) 

    return val[0], val[1]
