""" Kronecker structured conditionals """
from ..dispatch import dispatch, evoke
from ..utils.batch_utils import batch_over_module_types
from ..utils.utils import get_batch_type
from .marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_spatial_conditional_diagional, gaussian_spatial_conditional
from .matrix_ops import batched_block_diagional

# Import Types
from ..data import Data, Input
from ..approximate_posteriors import MeanFieldApproximatePosterior
from ..likelihood import Likelihood
from ..models import BatchGP, SDE_GP
from ..transforms import Independent

import jax
import jax.numpy as np
from jax import jit, vmap
import chex
import objax
from batchjax import batch_or_loop, BatchType


@dispatch(Data, Input, SDE_GP, 'GPPrior')
@dispatch(Data, Data, SDE_GP, 'GPPrior')
def spatial_conditional(XS_data: 'Data', X_data: 'Data', pred_mean, pred_var, gp, diagonal):
    """
    gp is a GP prior with a spatio-temporal kernel 

    Computes:
        mu = [ I ⊗ Ksz Kzz⁻¹ ] m
        var = diag[ Ktt ] ⊗ diag[ Kss - Ksz Kzz⁻¹ Kss] - diag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """
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

@dispatch(Data, Data, SDE_GP, Independent)
def spatial_conditional(XS: 'Data', X: 'Data', pred_mean, pred_var, gp, diagonal):
    """
    Let P be the number of outputs then:

    In:
        pred_mean: Nt x Ns*P x 1
        pred_var: Nt x Ns*P x Ns*P

    where pred_mean, pred_var are in latent-data format.    

    This treats each latent function separately
    """
    prior = gp.prior
    likelihood = gp.likelihood
    P = prior.num_latents
    latents_arr = prior.latents

    # TODO: check this
    Nt = pred_mean.shape[0]
    pred_mean = np.reshape(pred_mean, [Nt, P, -1])
    Ns = pred_mean.shape[-1]

    #Treat latents separately and ignore cross correlations
    pred_var = batched_block_diagional(pred_var, Ns)

    # Batch over latents
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'spatial_conditional',
        evoke_params = [XS, X, gp],
        module_arr = [latents_arr],
        fn_params = [XS, X, pred_mean, pred_var, latents_arr, diagonal],
        fn_axes = [None, None, 1, 1, 0, None],
        dim = len(latents_arr),
        out_dim  = 2
    )

    return marginal_mu, marginal_var

def _batched_st_kernel(X1, X2, prior, kernel_type='spatial', full=True):
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


def spatial_conditional_block(XS_space, X_space, Kzz, Ksz, Kss, Ktt, pred_mean, pred_var):
    """
    All inputs are in latent-data format either explictely or implictely.

    """
    breakpoint()
    pass

@dispatch(Data, Data, SDE_GP, Independent)
def spatial_conditional_block(XS_data: 'Data', X_data: 'Data', pred_mean, pred_var, gp, diagonal):
    """
    Let P be the number of outputs then:

    In:
        pred_mean: Nt x Ns*P x 1
        pred_var: Nt x Ns*P x Ns*P

    where pred_mean, pred_var are in latent-data format.    
    """
    prior = gp.prior
    likelihood = gp.likelihood
    P = prior.num_latents
    latents_arr = prior.latents

    XS_time = XS_data.X_time
    X_time = X_data.X_time

    # Get spatial locations with dummy time dimension so kernel evaluations are correct
    XS_space = XS_data.X_space
    X_space = X_data.X_space
    XS_space = np.hstack([np.zeros([XS_space.shape[0], 1]), XS_space])
    X_space = np.hstack([np.zeros([X_space.shape[0], 1]), X_space])

    # Precompute all kernels
    Ktt = _batched_st_kernel(X_time, X_time, prior, 'temporal', full=False)
    Kss = _batched_st_kernel(XS_space, XS_space, prior, 'spatial', full=True)
    Ksz = _batched_st_kernel(XS_space, X_space, prior, 'spatial', full=True)
    Kzz = _batched_st_kernel(X_space, X_space, prior, 'spatial', full=True)

    pred_mean = pred_mean[..., None]
    jax.vmap(
        spatial_conditional_block,
        [None, None, None, None, None, 1, 0, 0]
    )(
        XS_space, X_space, Kzz, Ksz, Kss, Ktt, pred_mean, pred_var
    )

    # Batch over latents
    marginal_mu, marginal_var = batch_over_module_types(
        evoke_name = 'spatial_conditional',
        evoke_params = [XS, X, gp],
        module_arr = [latents_arr],
        fn_params = [XS, X, pred_mean, pred_var, latents_arr, diagonal],
        fn_axes = [None, None, 1, 1, 0, None],
        dim = len(latents_arr),
        out_dim  = 2
    )

    return marginal_mu, marginal_var

@dispatch(Data, Data, SDE_GP, 'GPPrior')
def block_spatial_conditional(XS_data: 'Data', X_data: 'Data', pred_mean, pred_var, gp):
    block_size = gp.prior.num_latents

    chex.assert_rank([pred_mean, pred_var], [2, 3])

    # Add extra dim to ensure rank 2 after batching
    pred_mean = pred_mean[..., None]
    raise NotImplementedError()
