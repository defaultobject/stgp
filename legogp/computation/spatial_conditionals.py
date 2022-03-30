""" Kronecker structured conditionals """
from ..dispatch import dispatch, evoke
from ..utils.batch_utils import batch_over_module_types
from .marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_spatial_conditional_diagional

# Import Types
from ..data import Data
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


@dispatch(Data, Data, SDE_GP, 'GPPrior')
def spatial_conditional(XS_data: 'Data', X_data: 'Data', pred_mean, pred_var, gp, diagonal):
    """
    gp is a GP prior with a spatio-temporal kernel 
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
        raise NotImplementedError()


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

        return mu, var

@dispatch(Data, Data, SDE_GP, Independent)
def spatial_conditional(XS: 'Data', X: 'Data', pred_mean, pred_var, gp, diagonal):
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

    # TODO: check this
    Nt = pred_mean.shape[0]
    pred_mean = np.reshape(pred_mean, [Nt, P, -1])
    Ns = pred_mean.shape[-1]
    pred_var = np.reshape(pred_var, [Nt, P, Ns, Ns])

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

