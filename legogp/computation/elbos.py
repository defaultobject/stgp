from ..utils.utils import can_batch
from ..approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior
from ..transforms import Independent, LinearTransform, LMC
import jax
import jax.numpy as np
from jax import jit
import objax
import chex
from batchjax import batch_or_loop

from ..dispatch import dispatch, evoke
from .expected_log_likelihoods import precomputed_expected_log_likelihood 
from .kullback_leiblers import gaussian_kl
from .marginals import diagonal_marginal, whitened_diagonal_marginal

def precompute_diagonal_variational_primitives(X, prior, approximate_posterior):
    K_x_arr = prior.var(X)
    mean_xx_arr = prior.mean(X) 
    m_arr = approximate_posterior.m
    S_arr = approximate_posterior.S
    S_chol_arr = approximate_posterior.S_chol

    mean_zz_arr = batch_or_loop(
        lambda prior: prior.mean(prior.sparsity.Z)[0],
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_flag = can_batch(prior.latents)
    )

    # comptue K_zz, K_xz
    K_zz_arr = batch_or_loop(
        lambda prior: prior.kernel.K(prior.sparsity.Z, prior.sparsity.Z),
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_flag = can_batch(prior.latents)
    )

    K_xz_arr = batch_or_loop(
        lambda prior: prior.kernel.K(X, prior.sparsity.Z),
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_flag = can_batch(prior.latents)
    )

    return mean_xx_arr, mean_zz_arr, K_x_arr, K_xz_arr, K_zz_arr, m_arr, S_chol_arr, S_arr



def precompute_variational_primitives(X, prior, approximate_posterior):
    K_xx_arr = prior.covar(X, X)
    mean_xx_arr = prior.mean(X) 
    m_arr = approximate_posterior.m
    S_chol_arr = approximate_posterior.S_chol
    S_arr = approximate_posterior.S

    mean_zz_arr = batch_or_loop(
        lambda prior: prior.mean(prior.sparsity.Z)[0],
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_flag = can_batch(prior.latents)
    )

    # comptue K_zz, K_xz
    K_zz_arr = batch_or_loop(
        lambda prior: prior.kernel.K(prior.sparsity.Z, prior.sparsity.Z),
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_flag = can_batch(prior.latents)
    )

    K_xz_arr = batch_or_loop(
        lambda prior: prior.kernel.K(X, prior.sparsity.Z),
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_flag = can_batch(prior.latents)
    )

    return mean_xx_arr, mean_zz_arr, K_xx_arr, K_xz_arr, K_zz_arr, m_arr, S_chol_arr, S_arr

@dispatch(object, object, object, Independent, MeanFieldApproximatePosterior, object)
def elbo(
        X: np.ndarray, Y: np.ndarray, likelihood: list, prior: Independent, approximate_posterior: MeanFieldApproximatePosterior, inference: 'Variational'
):
    P = len(likelihood)
    Q = prior.num_outputs

    N = Y.shape[0]

    minibatch = False
    if inference.minibatch_size is not None:
        minibatch = True

    if minibatch:
        #minibatch
        minibatch_size = inference.minibatch_size

        idx = objax.random.randint((minibatch_size,), low=0, high=N-1, generator=inference.generator)

        X = X[idx,:]
        Y = Y[idx,:]

    else:
        minibatch_size = N

    # collect approx posteriors and prior latent functions
    prior_latents = prior.latents
    approx_posts = approximate_posterior.approx_posteriors

    sparsity_list = prior.get_sparsity_list()

    # precompute 
    mean_xx_arr, mean_zz_arr, K_xx_arr, K_xz_arr, K_zz_arr, m_arr, S_chol_arr, S_arr = precompute_diagonal_variational_primitives(X, prior, approximate_posterior)

    # Compute marginals q(f)
    # Collext all marginal dispatched callers
    # TODO: assume that all approx_posts are the same

    marginal_callers = [
        diagonal_marginal.dispatch(
            type(approx_posts[0]),
            type(sparsity_list[p])
        )
        for p in range(P)
    ]
    # assume all marginals callers are the same
    q_f_m_arr, q_f_s_arr = batch_or_loop(
        lambda X, mean, kxx, kxz, kzz, m, S, prior, q: marginal_callers[0](X, mean, kxx, kxz, kzz, m, S, prior.sparsity, q),
        [X, mean_zz_arr, K_xx_arr, K_xz_arr, K_zz_arr, m_arr, S_chol_arr, prior.latents, approx_posts],
        [None, 0, 0, 0, 0, 0, 0, 0, 0],
        dim = len(sparsity_list),
        out_dim = 2,
        batch_flag = can_batch(sparsity_list)
    )

    # Collect  all ELL functions
    ell_callers = [
        precomputed_expected_log_likelihood.dispatch(
            object, 
            object, 
            type(approx_posts[p]), 
            type(prior_latents[p]), 
            type(likelihood[p]), 
            object, 
            object
        )
        for p in range(P)
    ]


    # Assume they are all the same
    # TODO: group or loop here
    ell_arr = batch_or_loop(
            lambda X, Y, q, prior, lik, q_f_m, q_f_s: ell_callers[0](X, Y[:, None], q, prior, lik, q_f_m, q_f_s),
        [X, Y, approx_posts, prior_latents, likelihood, q_f_m_arr, q_f_s_arr],
        [None, 1, 0, 0, 0, 0, 0],
        dim = P,
        out_dim = 1,
        batch_flag = can_batch(likelihood)
    )

    # Compute KL
    kl_arr = batch_or_loop(
        gaussian_kl,
        [m_arr, S_arr, mean_zz_arr, K_zz_arr],
        [0, 0, 0, 0],
        dim = P,
        out_dim = 1,
        batch_flag = can_batch(mean_zz_arr)
    )

    val =  (N/minibatch_size) * np.sum(ell_arr) - np.sum(kl_arr)

    return val
