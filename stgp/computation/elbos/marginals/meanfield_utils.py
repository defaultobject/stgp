import chex
import jax
import jax.numpy as np
import objax
from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve, batched_block_diagional, to_block_diag
from ....core import Block, get_block_dim

def meanfield_marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten: bool, XS=None, sparsity=None, prediction=True):
    """ Helper function to collect marginals across a mean-field approximate posterior"""
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    latents_arr = prior.parent
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(latents_arr)
    N = data.X.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]

    # TODO: fix block sizes here
    out_block_arr = [likelihood_arr[0].block_type for q in range(num_latents)]

    N, Q, LB, _ = q_S_chol.shape
    N, QL, B = q_m.shape
    L = int(QL/Q)

    q_m = np.reshape(q_m, [N, Q, L, B])
    
    if not prediction:
        # Compute q(f) for each output
        # add additional dimension to q_m and q_S_chol to ensure rank [3, 4] after batching
        marginal_mu, marginal_var = batch_over_module_types(
            evoke_name = 'marginal_blocks',
            evoke_params = [],
            module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr],
            fn_params = [data, q_m, q_S_chol[:, :, None, ...], approx_posteriors_arr, likelihood_arr, latents_arr,  out_block_arr, whiten_arr],
            fn_axes = [None, 1, 1, 0, 0, 0, 0, 0],
            dim = len(latents_arr),
            out_dim  = 2,
            evoke_kwargs = {'whiten': whiten}
        )
    else:
        sparsity_arr = sparsity

        marginal_mu, marginal_var = batch_over_module_types(
            evoke_name = 'marginal_prediction_blocks',
            evoke_params = [],
            module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
            fn_params = [XS, data, q_m, q_S_chol[:, :, None, ...], approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
            fn_axes = [None, None, 1, 1, 0, 0, 0, 0, 0, 0],
            dim = len(latents_arr),
            out_dim  = 2,
            evoke_kwargs = {'whiten': whiten}
        )

    chex.assert_rank([marginal_mu, marginal_var], [4, 5])
    Q1, N1, P1, B1 = marginal_mu.shape
    Q2, N2, P2, B2, _ = marginal_var.shape

    # fix shapes
    #  each component will return rank (3, 4). stack into independent (block diagonal)
    # convert to N - (Q - P) - B
    marginal_mu = np.reshape(
        np.transpose(marginal_mu, [1, 0, 2, 3]), 
        [N1, Q1*P1, B1]
    )
    marginal_var = np.transpose(marginal_var, [1, 0, 2, 3, 4]) 
    marginal_var = marginal_var[:, :, 0, :, :]
    marginal_var = jax.vmap(to_block_diag)(marginal_var)
    marginal_var = marginal_var[:, None, ...]
    
    out_block_dim = get_block_dim(out_block_arr[0])

    #chex.assert_shape(marginal_mu, [N, num_latents,  out_block_dim])
    #chex.assert_shape(marginal_var, [N, num_latents, out_block_dim, out_block_dim])

    return marginal_mu, marginal_var

