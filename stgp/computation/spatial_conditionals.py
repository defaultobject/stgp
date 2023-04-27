""" Kronecker structured conditionals """
from ..dispatch import dispatch, evoke
from ..utils.batch_utils import batch_over_module_types
from ..utils.utils import get_batch_type
from .marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_spatial_conditional_diagional, gaussian_spatial_conditional, gaussian_linear_operator_spatial_conditional, gaussian_conditional_blocks, gaussian_linear_operator_spatial_conditional_blocks
from .matrix_ops import batched_block_diagional, to_block_diag, add_jitter, cholesky, get_block
from .permutations import permute_vec, permute_mat, data_order_to_output_order
from .. import settings 

# Import Types
from ..data import Data, Input
from ..approximate_posteriors import MeanFieldApproximatePosterior, FullGaussianApproximatePosterior,FullConjugateGaussian
from ..likelihood import Likelihood
from ..models import BatchGP, BASE_SDE_GP
from ..transforms import Independent, Joint
from ..transforms.pdes import DifferentialOperatorJoint
from ..transforms.sdes import SDE

import jax
import jax.numpy as np
from jax import jit, vmap, lax
import chex
import objax
from batchjax import batch_or_loop, BatchType


def _batched_diff_kernel(prior, X_time, XS_space, X_space, hierarchical):
    """ 
    Helper function to compute diff-op time-space kernels separately 

    Prior must be a GP with a spatio-temporal kernel.

    Requires the following format:

    Not Hierarchical:
        prior = Independent: [
            DifferentialOperatorJoint(
                DifferentialOperatorJoint(
                    kern = K_time * K_space,
                    derivative_kernel = Time Only
                )
                derivative_kernel = Space Only
            )
        ]
    """
    q_list = prior.parent

    # Helper function to batch across prior
    batch = lambda fn: batch_or_loop(
        fn,
        [q_list],
        [0],
        dim = len(q_list),
        out_dim = 1,
        batch_type = get_batch_type(q_list)
    )

    # returns p(S | T)
    base_prior = lambda q: q.base_prior

    # return K_time
    base_time_kernel = lambda q: base_kern(q).k1

    # return K_space
    base_space_kernel = lambda q: base_kern(q).k2

    Ms = X_space.shape[0]

    if hierarchical:
        # When hierarchical the base prior is directly the derivative kernel
        base_kern = lambda q: base_prior(q).derivative_kernel.parent_kernel

        # The spatial kernel on the prior does not contain derivaties
        # These are only included in the conditional

        # Ns x Ns
        K_base_spatial_zz_fn = lambda q: base_space_kernel(q).K(X_space, X_space)

        K_spatial_sz_fn = lambda q: (q.covar_from_fn(XS_space, X_space, base_space_kernel(q).K))[:, :Ms]

        K_x_t_fn = lambda q: jax.vmap(
            lambda t: base_prior(q).covar_from_fn(t, t, base_time_kernel(q).K)
        )(X_time[:, None, :])

        K_spatial_ss_fn = lambda q: jax.vmap(
            lambda x: q.covar_from_fn(x[None, :], x[None, :], base_space_kernel(q).K)
        )(XS_space)

    else:
        # return K_time * K_space
        base_kern = lambda q: base_prior(q).parent.derivative_kernel.parent_kernel

        # Compute \nabla K_space 
        K_base_spatial_zz_fn = lambda q: base_prior(q).covar_from_fn(X_space, X_space, base_space_kernel(q).K)

        K_spatial_sz_fn = lambda q: base_prior(q).covar_from_fn(XS_space, X_space, base_space_kernel(q).K) 

        K_x_t_fn = lambda q: jax.vmap(
            lambda t: base_prior(q).parent.covar_from_fn(t, t, base_time_kernel(q).K)
        )(X_time[:, None, :])

        K_spatial_ss_fn = lambda q: jax.vmap(
            lambda x: base_prior(q).covar_from_fn(x[None, :], x[None, :], base_space_kernel(q).K)
        )(XS_space)

    # N referes to the number of spatial points

    # [Q] x [Ds x N] x [Ds x N]
    K_base_spatial_zz_arr = batch(K_base_spatial_zz_fn)

    # [Q] x [Ds x Ns] x [Ds x N]
    K_spatial_sz_arr = batch(K_spatial_sz_fn)

    # [Q] x [Ns] x [Ds] x [Ds]
    K_base_spatial_ss_arr = batch(K_spatial_ss_fn)

    #[Q] x [Nt] x [Dt] x [Dt]
    K_x_t_arr= batch(K_x_t_fn)

    # Fix shapes
    # [Ns] x [Q] x [Ds] x [Ds]
    K_base_spatial_ss_arr = np.transpose(K_base_spatial_ss_arr, [1, 0, 2, 3])

    #[Nt] x [Q] x [ Dt] x [ Dt]
    K_x_t_arr = np.transpose(K_x_t_arr, [1, 0, 2, 3])

    return K_x_t_arr, K_base_spatial_ss_arr, K_spatial_sz_arr, K_base_spatial_zz_arr

def _batched_st_kernel(X1, X2, prior, kernel_type='spatial', full=True):
    """ 
    Helper function to compute time-space kernels separately 

    Prior must be a GP with a spatio-temporal kernel.
    """

    if kernel_type == 'spatial':
        if full:
            fn = lambda q: q.kernel.k2.K(X1, X2)
        else:
            fn = lambda q: q.kernel.k2.K_diag(X1)
    elif kernel_type == 'temporal':
        if full:
            fn = lambda q: q.kernel.k1.K(X1, X2)
        else:
            fn = lambda q: q.kernel.k1.K_diag(X1)
    else:
        raise NotImplementedError()

    # TODO: .latents is depreciated
    q_list = prior.latents

    K_arr = batch_or_loop(
        fn,
        [q_list],
        [0],
        dim = len(q_list),
        out_dim = 1,
        batch_type = get_batch_type(q_list)
    )

    return K_arr


def spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, prior):
    """
    Let P be the number of outputs then:

    In:
        pred_mean: Nt x P*Ns x 1
        pred_var: Nt x P*Ns x P*P

    where pred_mean, pred_var are in time-latent-space format.    


    Computes:
        mu = [ I ⊗ Ksz Kzz⁻¹ ] m
        var =  Ktt  ⊗ diag[ Kss - Ksz Kzz⁻¹ Kss] - diag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t

    """
    # TODO: assuming that prior is independent

    XS_time = data_xs.X_time
    X_time = data_x.X_time

    # Get spatial locations with dummy time dimension so kernel evaluations are correct
    XS_space = data_xs.X_space
    X_space = data_x.X_space
    XS_space = np.hstack([np.zeros([XS_space.shape[0], 1]), XS_space])
    X_space = np.hstack([np.zeros([X_space.shape[0], 1]), X_space])

    Ns = X_space.shape[0]
    Nss = XS_space.shape[0]

    # Precompute all kernels

    # TODO: how to compute the diagonal diff op kernels effeciently

    # latent - time format
    Ktt = _batched_st_kernel(XS_time, XS_time, prior, 'temporal', full=False)
    # latent - space format
    Kss = _batched_st_kernel(XS_space, XS_space, prior, 'spatial', full=True)
    # latent - space format
    Ksz = _batched_st_kernel(XS_space, X_space, prior, 'spatial', full=True)
    # latent - space format
    Kzz = _batched_st_kernel(X_space, X_space, prior, 'spatial', full=True)

    # in latent-space format
    Kss_full = to_block_diag(Kss)
    Ksz_full = to_block_diag(Ksz)
    Kzz_full = to_block_diag(Kzz)

    Q = Kzz.shape[0]

    # if the temporal kernel is a derivate kernel this will return a rank 3 matrix
    if len(Ktt.shape) == 2:
        f_only_flag: bool = True
    else:
        f_only_flag: bool = False

    if f_only_flag:
        # time - latent format
        Ktt = Ktt.T
        # time - latent - space format
        Ktt_full = jax.vmap(
            lambda _ktt: to_block_diag(jax.vmap(
                lambda _k: _k*np.ones([Nss, Nss]),
                0
            )(_ktt)),
            0
        )(Ktt)
    else:
        Ktt_full = Ktt[0]

    if False:
        if not f_only_flag:
            Kzz_full = Kzz_full[:Ns, ...][..., :Ns]
            Ksz_full = Ksz_full[..., :Ns]


    # TODO: check this
    mean_x = np.zeros([pred_mean.shape[1], 1])

    # prior.temporal_output_dim actually returns the state dim across all latents
    mean_xs = np.zeros([int((prior.temporal_output_dim/Q)) * Kss_full.shape[0], 1])

    # compute cholesky at each time stamp
    pred_var_chol = jax.vmap(
        lambda S: cholesky(add_jitter(S, settings.jitter)),
        0,
    )(pred_var)

    # batch over time

    if f_only_flag:
        spatial_fn = gaussian_spatial_conditional
    else:
        spatial_fn = gaussian_linear_operator_spatial_conditional

    # TODO: derive proper mean 

    if prior.whiten_space:
        # whiten transform in space
        Kzz_chol = cholesky(add_jitter(Kzz_full, settings.jitter))
        # TODO: fix the hardcoded time dim
        Kzz_chol = np.kron(np.eye(2), Kzz_chol)
        pred_mean, pred_var_chol =  Kzz_chol @ pred_mean, Kzz_chol @ pred_var_chol

    mu, var = jax.vmap(
        spatial_fn,
        [None, None, None, None, None, 0, 0, 0, None, None],
    )( 
        XS_space, 
        X_space, 
        Kzz_full, 
        Ksz_full, 
        Kss_full, 
        Ktt_full, #batching 
        pred_mean, #batching
        pred_var_chol, #batching
        mean_x, 
        mean_xs
    )

    # in time-latent-space format
    var = var[:, None, ...]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch(Data, Input, BASE_SDE_GP, 'GPPrior')
@dispatch(Data, Data, BASE_SDE_GP, 'GPPrior')
@dispatch(Data, Data, BASE_SDE_GP, SDE)
def spatial_conditional(data_xs: 'Data', data_x: 'Data', pred_mean, pred_var, gp, diagonal):
    """
    gp is a GP prior with a spatio-temporal kernel 

    Computes:
        mu = [ I ⊗ Ksz Kzz⁻¹ ] m
        var = diag[ Ktt ] ⊗ diag[ Kss - Ksz Kzz⁻¹ Kss] - diag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """
    mu, var = spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, gp.prior)
    return mu, var

@dispatch(Input, Independent, Independent, FullGaussianApproximatePosterior)
@dispatch(Data, Independent, Independent, FullGaussianApproximatePosterior)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    approximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):
    """
    Let P be the number of outputs then:

    In:
        pred_mean: Nt x Ns*P x 1
        pred_var: Nt x Ns*P x Ns*P

    where pred_mean, pred_var are in latent-data format.    
    """

    mu, var = spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, prior)
    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch(Data, Independent, DifferentialOperatorJoint, FullConjugateGaussian)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    aapproximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):
    """
    Posterior is in [time - Q - df - ds - space ] format
        we need it in [time - space - Q - df - ds] format
    """
    XS_time = data_xs.X_time

    # Data is the sparsity object
    #X_space = sparsity[0].raw_Z.X_space
    # Get spatial locations with dummy time dimension so kernel evaluations are correct
    XS_space = data_xs.X_space
    X_space = data_x.X_space
    X_time = data_xs.X_time

    space_dim = X_space.shape[1]
    Ns = data_x.Ns

    XS_space = np.hstack([np.zeros([XS_space.shape[0], 1]), XS_space])
    X_space = np.hstack([np.zeros([X_space.shape[0], 1]), X_space])

    X_time = np.hstack([X_time[:, None], np.zeros([X_time.shape[0], 1])])

    # TODO: assuming that data_xs and data_x are the same

    dummy_prior = prior.parent[0]
    hierarchical = dummy_prior.hierarchical

    # compute the outputs of each of the components of prior
    Q = len(prior.parent)
    base_prior_output = dummy_prior.base_prior.output_dim
    prior_added_output = dummy_prior.derivative_kernel.d_computed
    out_dim = base_prior_output * prior_added_output

    K_x_t, K_spatial_ss, K_spatial_sz, K_base_spatial_zz = _batched_diff_kernel(prior, X_time, XS_space, X_space, hierarchical)

    # compute cholesky at each time stamp
    pred_var_chol = jax.vmap(
        lambda S: cholesky(add_jitter(S, settings.jitter)),
        0,
    )(pred_var)


    # TODO: check this
    mean_x = np.zeros([pred_mean.shape[1], 1])
    mean_xs = np.zeros([data_xs.Ns * out_dim*Q, 1])

    # batch over time
    mu_p, var_p_bd = jax.vmap(
        gaussian_linear_operator_spatial_conditional_blocks,
        [None, None, None, None, None, None, 0, 0, 0, None, None],
    )( 
        Q*out_dim,
        XS_space, 
        X_space, 
        K_base_spatial_zz, 
        K_spatial_sz, 
        K_spatial_ss, 
        K_x_t, #batching 
        pred_mean, #batching
        pred_var_chol, #batching
        mean_x, 
        mean_xs
    )
    mu_p_bd = np.reshape(mu_p, [-1, Q* out_dim, 1])
    var_p_bd = np.reshape(var_p_bd, [-1, 1, Q*out_dim, Q*out_dim])

    chex.assert_rank([mu_p_bd, var_p_bd], [3, 4])


    return mu_p_bd, var_p_bd




@dispatch(Data, DifferentialOperatorJoint, DifferentialOperatorJoint, FullConjugateGaussian)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    aapproximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):
    """
    Single Latent Function case
    """
    breakpoint()
    # Wrap prior in an Independent transform so that it is in the same format as the multi latent case
    ind_prior = Independent([prior])
    
    return evoke('spatial_conditional', data, ind_prior, prior, approximate_posterior)(
           data_xs, data_x, pred_mean, pred_var, aapproximate_posterior, likelihood, ind_prior, sparsity, out_block_dim, whiten 
    )



