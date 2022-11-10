""" Kronecker structured conditionals """
from ..dispatch import dispatch, evoke
from ..utils.batch_utils import batch_over_module_types
from ..utils.utils import get_batch_type
from .marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_spatial_conditional_diagional, gaussian_spatial_conditional
from .matrix_ops import batched_block_diagional, to_block_diag, add_jitter, cholesky
from .. import settings 

# Import Types
from ..data import Data, Input
from ..approximate_posteriors import MeanFieldApproximatePosterior, FullGaussianApproximatePosterior
from ..likelihood import Likelihood
from ..models import BatchGP, BASE_SDE_GP
from ..transforms import Independent, Joint
from ..transforms.pdes import DifferentialOperatorJoint
from ..transforms.sdes import SDE

import jax
import jax.numpy as np
from jax import jit, vmap
import chex
import objax
from batchjax import batch_or_loop, BatchType


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

    Ns = XS_space.shape[0]

    # Precompute all kernels

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

    # time - latent format
    Ktt = Ktt.T
    # time - latent - space format
    Ktt_full = jax.vmap(
        lambda _ktt: to_block_diag(jax.vmap(
            lambda _k: _k*np.ones([Ns, Ns]),
            0
        )(_ktt)),
        0
    )(Ktt)

    # TODO: check this
    mean_x = np.zeros([Kzz_full.shape[0], 1])
    mean_xs = np.zeros([Kss_full.shape[0], 1])

    # compute cholesky at each time stamp
    pred_var_chol = jax.vmap(
        lambda S: cholesky(add_jitter(S, settings.jitter)),
        0,
    )(pred_var)

    # batch over time

    mu, var = jax.vmap(
        gaussian_spatial_conditional,
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

    chex.assert_rank([pred_mean, pred_var], [2, 3])

    # Add extra dim to ensure rank 2 after batching
    pred_mean = pred_mean[..., None]

    XS_time = XS_data.X_time
    X_time = X_data.X_time

    # Get spatial locations with dummy time dimension so kernel evaluations are correct
    XS_space = XS_data.X_space
    X_space = X_data.X_space
    XS_space = np.hstack([np.zeros([XS_space.shape[0], 1]), XS_space])
    X_space = np.hstack([np.zeros([X_space.shape[0], 1]), X_space])

    mean_x = np.zeros([X_space.shape[0], 1])
    mean_xs = np.zeros([XS_space.shape[0], 1])

    if diagonal:
        Ktt = gp.kernel.k1.K_diag(X_time)
        Kss = gp.kernel.k2.K_diag(XS_space)
    else:
        Ktt = gp.kernel.k1.K_diag(X_time)
        Kss = gp.kernel.k2.K(XS_space, XS_space)

    # Evaluate separable kernels
    Kzz = gp.kernel.k2.K(X_space, X_space)
    Ksz = gp.kernel.k2.K(XS_space, X_space)

    if diagonal:
        mu, var = jax.vmap(
            gaussian_spatial_conditional_diagional,
            [None, None, None, None, None, 0, 0, 0, None, None],
        )( 
            XS_space, X_space, Kzz, Ksz, Kss, Ktt, pred_mean, pred_var, mean_x, mean_xs
        )
    else:
        mu, var = jax.vmap(
            gaussian_spatial_conditional,
            [None, None, None, None, None, 0, 0, 0, None, None],
        )( 
            XS_space, X_space, Kzz, Ksz, Kss, Ktt, pred_mean, pred_var, mean_x, mean_xs
        )


    return mu, var

@dispatch(Input, Independent, FullGaussianApproximatePosterior)
@dispatch(Data, Independent, FullGaussianApproximatePosterior)
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


@dispatch(Input, DifferentialOperatorJoint, FullGaussianApproximatePosterior)
@dispatch(Data, DifferentialOperatorJoint, FullGaussianApproximatePosterior)
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
    XS_time = data_xs.X_time
    X_time = data_x.X_time

    # Get spatial locations with dummy time dimension so kernel evaluations are correct
    XS_space = data_xs.X_space
    X_space = data_x.X_space
    XS_space = np.hstack([np.zeros([XS_space.shape[0], 1]), XS_space])
    X_space = np.hstack([np.zeros([X_space.shape[0], 1]), X_space])

    # this will be premultiplied by Ktt
    #Kss = prior.covar(X_space, X_space)
    #Ktt = prior.

    # 
    #Kss = prior.parent.derivative_kernel.parent_kernel.K(X_space, X_space)

     

    breakpoint()


    mu, var = spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, prior)
    chex.assert_rank([mu, var], [3, 4])
    return mu, var
