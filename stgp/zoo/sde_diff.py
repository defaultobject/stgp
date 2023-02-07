import jax
import objax

import stgp
from stgp import settings
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32, SpatioTemporalSeperableKernel
from stgp.means.mean import FirstOrderDerivativeMean, SecondOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel, FirstOrderDerivativeKernel_2D, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel
from stgp.likelihood import Gaussian, BlockDiagonalGaussian, ProductLikelihood
from stgp.models import GP
from stgp.transforms import OutputMap
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.data import Data
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import  progress_bar_callback
from stgp.kernels.spectral_mixture import SM_Component
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian
from stgp.transforms import Independent
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer, ADAM
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE, LTI_SDE_Full_State_Obs_With_Mask

import numpy as onp

def get_time_diff_kernel_mean(time_kernel, time_diff):
    if time_diff == 1:
        time_kern = FirstOrderDerivativeKernel(time_kernel, input_index = 0)
        time_mean = FirstOrderDerivativeMean(parent_output_dim=1)
    elif time_diff == 2:
        time_kern = SecondOrderDerivativeKernel(time_kernel, input_index = 0)
        time_mean = SecondOrderDerivativeMean(parent_output_dim=1)

    return time_kern, time_mean

def get_space_diff_kernel_mean(space_kernel, space_diff, time_diff_kern = None):
    if space_diff == 1:
        if time_diff_kern is None:
            # for surrogate SDE
            space_kern = FirstOrderDerivativeKernel(space_kernel, input_index = 1)
            space_mean = FirstOrderDerivativeMean(input_index=1)
        else:
            # for vi model
            space_kern = FirstOrderDerivativeKernel(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim)
            space_mean = FirstOrderDerivativeMean(input_index=1, parent_output_dim=time_diff_kern.output_dim)

    elif space_diff == 2:
        if time_diff_kern is None:
            space_kern = SecondOrderDerivativeKernel(space_kernel, input_index = 1)
            space_mean = SecondOrderDerivativeMean(input_index=1)
        else:
            raise NotImplementedError()
    elif space_diff == -2:
        if time_diff_kern is None:
            space_kern = SecondOrderOnlyDerivativeKernel(space_kernel, input_index = 1)
            space_mean = SecondOrderOnlyDerivativeKernel(input_index=1)
        else:
            raise NotImplementedError()

    return space_kern, space_mean

def diff_sparse_sde_vgp(X, Y, time_diff = 1, space_diff = 1, time_kernel = None, space_kernel = None, fix_y=False, lik_var = 1.0, Z= None, train_Z = True, ell_samples=None, prior_fn = None, keep_dims=None):

    if time_kernel is None:
        raise RuntimeError('Time Kernel must be passed!')

    if space_kernel is None:
        raise RuntimeError('Space Kernel must be passed!')


    if Z is None:
        raise RuntimeError('Z must be passed!')

    include_space = not(space_kernel is None)

    N, P = Y.shape
    Ms = Z.shape[0]

    # Construct Space-time data and kernels
    data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
    #base_kernel = SpatioTemporalSeperableKernel(time_kernel,  space_kernel)
    base_kernel = time_kernel*space_kernel

    if prior_fn is None:
        lik_arr = [Gaussian(lik_var) for p in range(P)]
    else:
        lik_arr = [ProductLikelihood([Gaussian(lik_var)]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    # construct time kernel
    if time_diff is not None:
        time_diff_kern, time_diff_mean = get_time_diff_kernel_mean(time_kernel, time_diff)
        hierarchical_diff_kern, hierarchical_diff_mean = get_time_diff_kernel_mean(base_kernel, time_diff)
        time_output_dim = time_diff_kern.output_dim
    else:
        time_output_dim = 1

    # construct space kernel
    space_diff_kern, space_diff_mean = get_space_diff_kernel_mean(space_kernel, space_diff)
    hierarchical_space_diff_kern, hierachical_space_diff_mean = get_space_diff_kernel_mean(space_kernel, space_diff, time_diff_kern)


    # construct surrogate SDE model
    # pass through the derivatie kernels as we want the SDE model to compute all the derivates
    # ie this is not the hierarchical model
    base_diff_kernel = SpatioTemporalSeperableKernel(
        time_diff_kern, 
        space_diff_kern,
        spatial_output_dim = space_diff_kern.output_dim
    )

    Z_sparsity = stgp.sparsity.SpatialSparsity(data.X_time, Z, train=train_Z)

    # these kernels will not be used really
    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=Z_sparsity, 
            kernel = base_kernel
        ),
        kernel = hierarchical_diff_kern,
        is_base = True,
        has_parent=False,
        hierarchical=False
    )

    # construct P(S | T)
    # even though we are not in a hierarchical model we have to set up the prior properly so that we can predict
    diff_op_prior = DifferentialOperatorJoint(
        diff_op_prior_time,
        kernel = space_diff_kern,
        mean = space_diff_mean,
        is_base = True,
        has_parent=True,
        hierarchical=False
    )


    # surrogate model prior
    latent_sde_gp = GP(
        sparsity=Z_sparsity, 
        kernel = base_diff_kernel
    )

    latent_sde_gp = Independent([latent_sde_gp])

    if keep_dims is None:
        latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp)
    else:
        latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(latent_sde_gp, keep_dims=keep_dims)


    Q = diff_op_prior.output_dim
    Q = 4
    B = Ms * Q
    q = FullConjugateGaussian(
        X = Z_sparsity,
        num_latents =  Q,
        block_size= B,
        num_blocks = data.Nt,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            # in state-space format
            data = stgp.data.SpatioTemporalData(X=X.raw_Z, Y=onp.reshape(Y, [data.Nt,  Q, Ms]), sort=False, train_y=True), # we need gradients Y so set to be trainable
            likelihood=likelihood, 
            prior=latent_sde_gp,
            inference='Sequential',
            full_state_observed = True
        )
    )


    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    # Create Model
    m = stgp.models.GP(
        data = data,
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q,
        ell_samples=ell_samples
    )


    return m
