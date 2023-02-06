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
from ....transforms.latent_variable import LatentVariable
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateApproximatePosterior, FullConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior
from ....sparsity import SpatialSparsity, NoSparsity

from .linear_marginals import linear_marginal_blocks

@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, NoSparsity, whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    if not prior.hierarchical:
        breakpoint()
        chex.assert_rank([q_m, q_S], [3, 4])
        chex.assert_equal(q_S.shape[1], 1)

        if (out_block == Block.FULL or out_block == Block.BLOCK):
            P = q_m.shape[1]
            Q = prior.output_dim

            mu_p = jax.vmap(lambda a: permute_vec(a, Q))(q_m)
            var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(q_S)

            var_p = var_p[:, None, ...]
            return mu_p, var_p
        else:
            if True:
                # q_m is in time - latent - space format
                N = data.N
                Nt, _, _= q_m.shape
                Q = prior.output_dim

                mu_p = jax.vmap(lambda a: permute_vec(a, Q))(q_m)
                var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(q_S)

                # extract block diagonals
                mu_p_bd = np.reshape(mu_p, [-1, Q, 1])
                var_p_bd = batched_block_diagional(var_p, Q)
                var_p_bd = np.reshape(var_p_bd, [-1, 1, Q, Q])

                chex.assert_rank([mu_p_bd, var_p_bd], [3, 4])
                return mu_p_bd, var_p_bd

            #assert out_block_dim == 1
            out_block_dim = 1

            # q_m is in time - latent - space format
            N = data.N
            Nt, P, Ns = q_m.shape

            # convert to data-latent format
            mu = np.reshape(np.transpose(q_m, [0, 2, 1]), [N, P, 1])
            var_p = jax.vmap(lambda A: permute_mat(A[0], P))(q_S)
            var = batched_block_diagional(var_p, P)

            chex.assert_rank([mu, var], [3, 4])
            return mu, var
    else:
        # compute spatial conditonal
        sparsity =  prior.base_prior.get_sparsity_list()

        out_block_dim = 1
        mu, var = evoke('spatial_conditional', data, prior, approximate_posterior)(
            data, 
            sparsity[0].raw_Z, 
            q_m, 
            q_S[:, 0, ...], 
            approximate_posterior,
            likelihood,
            prior,
            sparsity,
            out_block_dim,
            whiten
        )

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, SpatialSparsity, whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    if not prior.hierarchical:
        if out_block == Block.LATENT:
            raise NotImplementedError()
            out_block_dim = 1
            mu, var = evoke('spatial_conditional', data, prior, approximate_posterior)(
                data, 
                sparsity[0].raw_Z, 
                q_m, 
                q_S[:, 0, ...], 
                approximate_posterior,
                likelihood,
                prior,
                sparsity,
                out_block_dim,
                whiten
            )

            chex.assert_rank([mu, var], [3, 4])

            return mu, var
        else:
            P = q_m.shape[1]
            Q = prior.output_dim
            Q = 4

            mu_p = jax.vmap(lambda a: permute_vec(a, Q))(q_m)
            var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(q_S)

            var_p = var_p[:, None, ...]
            return mu_p, var_p
    else:
        # compute spatial conditonal
        sparsity =  prior.base_prior.get_sparsity_list()

        out_block_dim = 1
        mu, var = evoke('spatial_conditional', data, prior, approximate_posterior)(
            data, 
            sparsity[0].raw_Z, 
            q_m, 
            q_S[:, 0, ...], 
            approximate_posterior,
            likelihood,
            prior,
            sparsity,
            out_block_dim,
            whiten
        )

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

# DifferentialOperatorJoint with CVI approximate posteriors
@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, out_block: Block, whiten: bool):
    sparsity =  prior.base_prior.get_sparsity_list()

    # handle the CVI marginal based on sparsity
    mu, var =  evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity[0], whiten=whiten, debug=True)(
        data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

# DifferentialOperatorJoint with (non-CVI) approximate posteriors
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: int, whiten: bool):
    """
    When we are not in the CVI setting we do not make any distinction between space and time derivates.

    We we are not hierarchical then q(U) is defined over all of the derivavates and can reuse all the marginals from the multi-output setting.

    If we are hierarchical then we compute q(u) and construct q(F) as

        q(F) = \int p(F | u) q(u) \du

    where

        p(F | u) = N( F | K^∇ K^∇_Z u, .)

    which is the same form as the standard required marginals just with a `multi-output' kernel.
    """
    if not prior.hierarchical:
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        base_prior = get_permutated_prior(prior)

        fn = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, base_prior, sparsity_type, whiten=whiten, debug=False)

        mu, var = fn(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten)

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

    else:
        chex.assert_rank([q_m, q_S_chol], [3, 4])

        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        Z = sparsity_arr[0].Z
        Q = prior.derivative_kernel.output_dim

        NS = XS.shape[0]
        M = Z.shape[0]

        # compute multi-output kernels

        # latent-data format
        # due to a jit bug we have to use the full gaussian_conditional
        #   but we only care about the diagonal of the result so we just compute
        #   the diagonal variance here

        #Kxx = np.diag(np.squeeze(prior.var(XS))) # ND x ND
        Kxx = prior.covar(XS, XS) # ND x ND
        mean_xx = prior.mean(XS)

        base_prior_output = prior.base_prior.output_dim
        prior_added_output = prior.derivative_kernel.d_computed

        # covar is ordered by K ⊗ D
        Kzz = prior.base_prior.covar(Z, Z)
        mean_zz = prior.base_prior.mean(Z)

        Kxz = prior.covar(XS, Z)

        # remove added outputs to Kxz Z dimension as Kzz is only defined on the base prior
        idx = np.hstack([np.arange((M*prior_added_output)*d, (M*prior_added_output)*d + M) for d in range(base_prior_output)])
        Kxz = Kxz[:, idx]

        # compute standard marginals
        if whiten:
            mu, var = whitened_gaussian_conditional_full(
                XS, 
                Z, 
                Kzz, 
                Kxz, 
                Kxx, 
                q_m[..., 0],
                q_S_chol[0, 0, ...]
            )
            # TODO: why the transpose?
            P = data_order_to_output_order(NS, prior.output_dim)
            post_mu = np.reshape(P.T @ mu, [NS, prior.output_dim, 1])

            post_var = P.T @ var @ P
            post_var = get_block_diagonal(post_var, prior.output_dim)
            post_var = np.reshape(post_var, [NS, 1, prior.output_dim, prior.output_dim])

            return post_mu, post_var
        else:
            # TODO: this is inefficiently implemented but it works 
            # TODO: why not just vmap over x? this will automatically get the correct format, and may even make the graph smaller?
            P = data_order_to_output_order(NS, prior.output_dim)

            Kxx_diag = jax.vmap(lambda x: prior.covar(x, x))(XS[:, None, ...])

            mu, var = gaussian_conditional_blocks(
                1.0,
                prior.output_dim,
                XS, 
                Z, 
                Kzz, 
                P.T @ Kxz, 
                Kxx_diag, 
                q_m[..., 0],
                q_S_chol[0, 0, ...], 
                mean_zz, 
                mean_xx
            )
           
            post_mu = mu[..., None]
            post_var = var[:, None, ...]
            return post_mu, post_var



# DifferentialOperatorJoint with (non-CVI) approximate posteriors
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):
    """
    DifferentialOperatorJoint with (non-CVI) approximate posteriors.

    If we are hierarchical then we need to compute the required derivatives so we call the necessary diff-op predictions.

    If we are not hierarchical,  q(F) is defined over all the required derivatives, and we are in the standard multi-output setting
        hence we just call the standard marginals
    """
    if prior.hierarchical:
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        mu, var =  evoke('marginal_prediction_blocks', approximate_posterior, likelihood, prior, sparsity_type, whiten=whiten)(
            data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity_arr, out_block, whiten
        )

        chex.assert_rank([mu, var], [3, 4])
        return mu, var
    else:
        # Call standard multi-output marginals
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        base_prior = get_permutated_prior(prior)

        fn = evoke('marginal_blocks', approximate_posterior, likelihood, base_prior, sparsity_type, whiten=whiten, debug=False)

        mu, var = fn(data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten)

        chex.assert_rank([mu, var], [3, 4])
        return mu, var


 
