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

from .linear_marginals import linear_marginal_blocks

# DifferentialOperatorJoint with CVI approximate posteriors
@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, out_block: Block, whiten: bool):
    if not prior.hierarchical:
        chex.assert_rank([q_m, q_S], [3, 4])
        chex.assert_equal(q_S.shape[1], 1)

        if out_block == Block.FULL or out_block == Block.BLOCK:
            return q_m, q_S
        else:
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

        if False:
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

        X_t = data.X_time  
        X_s = data.X_space


        X_i = np.hstack([np.tile(X_t[0:1][:, None], [X_s.shape[0], 1]), X_s])

        # compute for all time slices
        Kzz = prior.parent.covar(X_i, X_i)
        # Ktt is ordered by [f, f_t]
        # Kxx is ordered by [f, f_t, f_s, f_ts]
        Kxx = prior.covar(X_i, X_i)

        M = X_i.shape[0]
        base_prior_output = prior.parent.output_dim
        prior_added_output = prior.derivative_kernel.d_computed

        idx = np.hstack([np.arange((M*prior_added_output)*d, (M*prior_added_output)*d + M) for d in range(base_prior_output)])
        Kxz = Kxx[:, idx]

        mean_zz = prior.parent.mean(X_i)
        mean_xx = prior.mean(X_i)

        q_m_t = q_m[0][:, None]
        q_S_t = cholesky(add_jitter(q_S[0], settings.jitter))

        mu, var = gaussian_conditional(
            X_i, 
            X_i, 
            Kzz, 
            Kxz, 
            Kxx, 
            q_m_t,
            q_S_t, 
            mean_zz, 
            mean_xx
        )

        # TODO FIGURE OUT ORDERING
        breakpoint()



        breakpoint()
        raise NotImplementedError()

# DifferentialOperatorJoint with (non-CVI) approximate posteriors
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: int, whiten: bool):
    if not prior.hierarchical:

        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        base_prior = get_permutated_prior(prior)

        fn = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, base_prior, sparsity_type, whiten=whiten, debug=True)

        mu, var = fn(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten)

        breakpoint()
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

        if False:
            # compute marginal q(f) \int p(f | u) q(u) df 
            fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior.base_prior, whiten=whiten)
            q_m, q_S = fn(
                data, q_m, q_S_chol, approximate_posterior, likelihood, prior.base_prior, out_block, whiten
            ) 
            chex.assert_rank([q_m, q_S], [3, 4])

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

            mu, var = gaussian_conditional(
                XS, 
                Z, 
                Kzz, 
                Kxz, 
                Kxx, 
                q_m[..., 0],
                q_S_chol[0, 0, ...], 
                mean_zz, 
                mean_xx
            )
            # TODO: why the transpose?
            P = data_order_to_output_order(NS, prior.output_dim)
            post_mu = np.reshape(P.T @ mu, [NS, prior.output_dim, 1])

            post_var = P.T @ var @ P
            post_var = get_block_diagonal(post_var, prior.output_dim)
            post_var = np.reshape(post_var, [NS, 1, prior.output_dim, prior.output_dim])

            return post_mu, post_var



# DifferentialOperatorJoint with (non-CVI) approximate posteriors
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):
    if not prior.hierarchical:
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        base_prior = get_permutated_prior(prior)

        fn = evoke('marginal_blocks', approximate_posterior, likelihood, base_prior, sparsity_type, whiten=whiten)

        mu, var = fn(data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten)

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

    else:
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]


        mu, var =  evoke('marginal_prediction_blocks', approximate_posterior, likelihood, prior, sparsity_type, whiten=whiten)(
            data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity_arr, out_block, whiten
        )

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

 
