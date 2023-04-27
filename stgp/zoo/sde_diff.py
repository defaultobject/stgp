"""
We use the convention that a negative diff means to compute onl that corresponding derivative:
    ie a diff of 2 computes [f, df/dt, df2/dt^2] where as -2 only computes [f, df2/dt^2]
"""
import jax
import jax.numpy as np
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
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian, MeanFieldConjugateGaussian
from stgp.transforms import Independent
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer, ADAM
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE, LTI_SDE_Full_State_Obs_With_Mask

import numpy as onp



def _get_time_diff_kernel_mean(time_kernel, time_diff):
    if time_diff == 1:
        time_kern = FirstOrderDerivativeKernel(time_kernel, input_index = 0)
        time_mean = FirstOrderDerivativeMean(parent_output_dim=1)
    elif time_diff == 2:
        time_kern = SecondOrderDerivativeKernel(time_kernel, input_index = 0)
        time_mean = SecondOrderDerivativeMean(parent_output_dim=1)
    elif time_diff == -2:
        time_kern = SecondOrderOnlyDerivativeKernel(time_kernel, input_index = 0)
        time_mean = SecondOrderOnlyDerivativeKernel(parent_output_dim=1) # not used

    return time_kern, time_mean

def get_time_diff_kernel_mean(time_kernel, time_diff):
    Q = len(time_kernel)
    res = [
        _get_time_diff_kernel_mean(time_kernel[q], time_diff)
        for q in range(Q)
    ]
    
    return [res[q][0] for q in range(Q)], [res[q][1] for q in range(Q)]


def _get_space_diff_kernel_mean(space_kernel, space_diff, time_diff_kern = None):
    """"
    There are two situations when creating a spatial diff kernel:
        
    Composition:
        The time kernel is passed to the space kernel and both the time and space outputs are computed together

    Heirarchical:
        The time kernel is not passed as the spatial part is computed separtely from the temporal part
    """
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
            space_mean = SecondOrderDerivativeMean(input_index=1) # not used
        else:
            space_kern = SecondOrderOnlyDerivativeKernel(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim)
            space_mean = SecondOrderDerivativeMean(input_index=1, parent_output_dim=time_diff_kern.output_dim) # not used

    return space_kern, space_mean

def get_space_diff_kernel_mean(space_kernel, space_diff, time_diff_kern = None):
    Q = len(space_kernel)

    if time_diff_kern is None:
        time_diff_kern = [None for q in range(Q)]

    res = [
        _get_space_diff_kernel_mean(space_kernel[q], space_diff, time_diff_kern[q])
        for q in range(Q)
    ]
    
    return [res[q][0] for q in range(Q)], [res[q][1] for q in range(Q)]



def diff_cvi_sde_vgp(
    X, Y, num_latents=None, time_diff = 1, space_diff = 1, time_kernel = None, space_kernel = None, space_diff_kernel = None, fix_y=False, lik_var = 1.0, Zs= None, train_Z = True, ell_samples=None, prior_fn = None, keep_dims=None , hierarchical=None, meanfield=False, parallel = False, multioutput_prior = False, verbose=False
):
    """
    Args:
        num_latents [None | int] - Number of latent multi-variate GPs
        time_diff: [int] - number of temporal diffs to compute - will be the same across all latents
        space_diff: [int] - number of spatial diffs to compute - will be the same across all latents
        time_kernel: [list[kernel]|kernel]
        space_kernel: [list[kernel]|kernel]
        space_diff_kernel: Optional[]  - optional pre-computed space diff kernel. Useful for passing a closed form. Only works for hierarchial
        lik_var: [float|list[float]] - Gaussian likelihood noise. If a list if not passed the same value is initialised acrossed all outputs
        Z_s: [None|np.ndarray|list[np.ndarray]] - Optional spatial inducing points. If passed then run in a sparse setting. 
        ell_samples: [None|int] - Optional number of monte-carlo samples to appoximate the ELL with
        prior_fn: [None|callable] - Optional function to transform the prior with
        keep_dims: [None, list[int]] - Optional dims of the state-space state to observe. Useful when using higher order states corresponding to Matern52/72 etc.
        hierarchical: [Optional[bool]] - Optional flag. If true then construct prior over temporal derivates and push spatial ones into the marginal/likelihood.
        meanfield: [bool] - default false. Construct a meanfield approximate posterior across the latents. Default is a full Gaussian.
        multioutput_prior[bool] - when multioutput the likelihood must be constructed as a list of productlikelihoods
        parallel: [bool] - default false. Whether or not use a parallel kalman filter and smoother.
    """

    if space_diff_kernel is not None:
        if not hierarchical:
            raise RuntimeError('Can only pass space_diff_kernel in a hierarchical model')

    # Figure out what setting we are constructing a model in
    dim = X.shape[1]

    if dim > 1:
        include_space = True
    else:
        include_space = False

    if num_latents is None:
        if type(time_kernel) is list:
            num_latents = len(time_kernel)
        else:
            num_latents = 1

    # Convert to multi-latent form
    if type(time_kernel) is not list:
        time_kernel = [time_kernel]
        space_kernel = [space_kernel]

    # Setup sequential data
    N, P = Y.shape
    if include_space:
        data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
    else:
        data = stgp.data.MultiOutputTemporalData(X=X, Y=Y, sort=True)
    Ms = data.Ns

    # We only support the same inducing locations across all latent functions
    #   this is due to how the multi-latent kalman filter is constructed

    if Zs is not None:
        if not include_space:
            raise RuntimeError('Cannot have spatial inducing points in the 1D setting')

        if type(Zs) is list:
            raise RuntimeError('We only support the same inducing locations across latents')
        sparsity = stgp.sparsity.SpatialSparsity(data.X_time, Zs, train=train_Z)
        sparse = True
    else:
        # If Z not passed assume NoSparsity
        # Pass X from data so that it will be  properly sorted
        sparsity = stgp.sparsity.NoSparsity(Z_ref=data._X)
        sparse = False

    # Setup Prior
    if include_space:
        base_kernel = [
            time_kernel[q] * space_kernel[q]
            for q in range(num_latents)
        ]
    else:
         base_kernel = time_kernel

    # All models have time so we can directly construct the time prior

    # construct time kernel
    if time_diff is not None:
        time_diff_kern, time_diff_mean = get_time_diff_kernel_mean(time_kernel, time_diff)
        composite_time_diff_kern, composite_time_diff_mean = get_time_diff_kernel_mean(base_kernel, time_diff)
    else:
        raise NotImplementedError()

    if include_space:
        # construct space kernel
        # we do not pass time as we are using a kalman filter which computes the spatial and temporal kernels separetely
        if space_diff_kernel is None:
            space_diff_kern, space_diff_mean = get_space_diff_kernel_mean(space_kernel, space_diff)
        else:
            space_diff_kern = space_diff_kernel
            space_diff_mean = [None for i in range(space_diff)] # not used atm so just create nans

        # in the composite case we  pass through the time_diff_kernel as want to compute something like
        #    kernel = FirstOrderDerivativeKernel(
        #        FirstOrderDerivativeKernel(base_kerns[i], input_index=0), 
        #        input_index=1, parent_output_dim = 2
        #    )
        composite_space_diff_kern, composite_space_diff_mean = get_space_diff_kernel_mean(composite_time_diff_kern, space_diff, composite_time_diff_kern)


    # set up base prior
    if hierarchical or sparse:
        diff_op_prior_time = [
            DifferentialOperatorJoint(
                GP(
                    sparsity=sparsity, 
                    kernel = base_kernel[q]
                ),
                # we need to pass the composite kernel as internally this kernel is used to access the spatial kernel
                kernel = composite_time_diff_kern[q], 
                is_base = True,
                has_parent=False,
                hierarchical=False
            )
            for q in range(num_latents)
        ]

        if include_space:
            # construct P(S | T)
            diff_op_prior = [
                DifferentialOperatorJoint(
                    diff_op_prior_time[q],
                    kernel = space_diff_kern[q],
                    mean = None,
                    is_base = True,
                    has_parent=True,
                    hierarchical=hierarchical
                )
                for q in range(num_latents)
            ]
        else:
            diff_op_prior = diff_op_prior_time

    else:
        if include_space:
            diff_op_prior = [
                DifferentialOperatorJoint(
                    GP(
                        sparsity=sparsity, 
                        kernel = base_kernel[q]
                    ),
                    kernel = composite_space_diff_kern[q],
                    is_base = True,
                    has_parent = False
                )
                for q in range(num_latents)
            ]
        else:
            diff_op_prior = [
                DifferentialOperatorJoint(
                    GP(
                        sparsity=sparsity, 
                        kernel = base_kernel[q]
                    ),
                    kernel = composite_time_diff_kern[q],
                    is_base = True,
                    has_parent = False
                )
                for q in range(num_latents)
            ]

    diff_op_prior = Independent(diff_op_prior)

    # setup surrogate SDE prior

    if include_space:
        if hierarchical:
            # when hierachical we do not compute the spatial derivates using the filter
            base_st_kerns = [
                SpatioTemporalSeperableKernel(
                    time_diff_kern[q], 
                    space_kernel[q]
                )
                for q in range(num_latents)
            ]
        else:
            base_st_kerns = [
                SpatioTemporalSeperableKernel(
                    time_diff_kern[q], 
                    space_diff_kern[q],
                    spatial_output_dim = space_diff_kern[q].d_computed
                )
                for q in range(num_latents)
            ]
    else:
        base_st_kerns = time_diff_kern

    # surrogate model prior
    if meanfield:
        if keep_dims is None:
            latent_sde_gp = Independent([
                LTI_SDE_Full_State_Obs(
                    Independent([
                        GP(
                            sparsity=sparsity, 
                            kernel = base_st_kerns[i]
                        )
                    ])
                )
                for i in range(num_latents)
            ])
        else:
            latent_sde_gp = Independent([
                LTI_SDE_Full_State_Obs_With_Mask(
                    Independent([
                        GP(
                            sparsity=sparsity, 
                            kernel = base_st_kerns[i]
                        )
                    ]),
                    keep_dims=keep_dims
                )
                for i in range(num_latents)
            ])

    else:
        if keep_dims is None:
            latent_sde_gp = LTI_SDE_Full_State_Obs(
                Independent([
                    GP(
                        sparsity=sparsity, 
                        kernel = base_st_kerns[q]
                    )
                    for q in range(num_latents)
                ])
            )
        else:
            latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(
                Independent([
                    GP(
                        sparsity=sparsity, 
                        kernel = base_st_kerns[q]
                    )
                    for q in range(num_latents)
                ]),
                keep_dims=keep_dims
            )


    # Setup likelihood
    if type(lik_var) is not list:
        lik_var = [lik_var for p in range(P)]

    # setup approximate posterior

    # When using keep_dims  only a subset of the full kalman state will be observed
    #    and the shape of Y and the likelihood only needs to be defined across the observed ones
    if keep_dims:
        state_dim = len(keep_dims)
    else:
        state_dim = time_kernel[0].state_space_dim()

    if include_space:
        if not hierarchical:
            # when not hierarchical 
            state_dim =  state_dim * space_diff_kern[0].output_dim

    if include_space:
        if sparse:
            Ms = sparsity.raw_Z.Ns
        else:
            Ms = data._X.Ns
    else:
        Ms = 1

    if meanfield:
        if include_space:
            if verbose:
                print(f'Q: {num_latents}, state_dim: {state_dim}, Nt: {data.Nt}, Ns: {data.Ns}, Ms: {Ms}')

            # ====== SPATIO-TEMPORAL MEANFIELD =======
            q = MeanFieldConjugateGaussian(
                approximate_posteriors = [
                    FullConjugateGaussian(
                        X = sparsity,
                        num_latents = state_dim,
                        block_size= Ms * state_dim,
                        num_blocks = data.Nt,
                        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                            # in state-space format
                            data = stgp.data.SpatioTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim, Ms]), sort=False),
                            likelihood=likelihood, 
                            prior=latent_sde_gp.parent[q],
                            inference='Sequential',
                            parallel=parallel,
                            full_state_observed=True
                        )
                    )
                    for q in range(num_latents)
                ]
            )
        else:
            # ====== TEMPORAL MEANFIELD =======
            q = MeanFieldConjugateGaussian(
                approximate_posteriors = [
                    FullConjugateGaussian(
                        X = sparsity,
                        num_latents = state_dim,
                        block_size= Ms * state_dim,
                        num_blocks = data.Nt,
                        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                            # in state-space format
                            data = stgp.data.MultiOutputTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim, Ms]), sort=False),
                            likelihood=likelihood, 
                            prior=latent_sde_gp.parent[q],
                            inference='Sequential',
                            parallel=parallel,
                            full_state_observed=True
                        )
                    )
                    for q in range(num_latents)
                ]
            )
    else:

        if include_space:
            if verbose:
                print(f'Q: {num_latents}, state_dim: {state_dim}, Nt: {data.Nt}, Ns: {data.Ns}, Ms: {Ms}')

            q = FullConjugateGaussian(
                X = sparsity,
                num_latents =  num_latents * state_dim,
                block_size= Ms * state_dim * num_latents,
                num_blocks = data.Nt,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    # in state-space format
                    data = stgp.data.SpatioTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim * num_latents, Ms]), sort=False),
                    likelihood=likelihood, 
                    prior=latent_sde_gp,
                    inference='Sequential',
                    parallel=parallel,
                    full_state_observed=True
                )
            )
        else:
            q = FullConjugateGaussian(
                X = sparsity,
                num_latents =  num_latents * state_dim,
                block_size= Ms * state_dim * num_latents,
                num_blocks = data.Nt,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    # in state-space format
                    data = stgp.data.MultiOutputTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim * num_latents, Ms]), sort=False),
                    likelihood=likelihood, 
                    prior=latent_sde_gp,
                    inference='Sequential',
                    parallel=parallel,
                    full_state_observed=True
                )
            )

    # Setup Prior Transform
    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    if multioutput_prior:
        lik_arr = [ProductLikelihood([Gaussian(lik_var[p])]) for p in range(P)]
    else:
        lik_arr = [Gaussian(lik_var[p]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()


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
