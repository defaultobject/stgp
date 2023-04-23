import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp

import objax

import numpy as np
import pandas as pd

import stgp
from stgp import settings
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.trainers.standard import VB_NG_ADAM, NatGradTrainer


from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32, SpatioTemporalSeperableKernel, ScaledMatern72

from stgp.means.mean import FirstOrderDerivativeMean, SecondOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel, FirstOrderDerivativeKernel_2D, SecondOrderDerivativeKernel
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms import Independent
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.data import Data
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE, LTI_SDE_Full_State_Obs_With_Mask

from stdata.grids import create_spatial_grid
from stdata.plots import grid_to_matrix

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/ohamelijnck/Documents/projects/stgp/examples')
from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors

stgp.settings.jitter = 1e-7

ls = 1.0
noise = 0.01

def get_im(X, Y):
    df = pd.DataFrame(X, columns=['lat', 'lon'])
    return grid_to_matrix(df, Y)

def get_data(X):
    N = X.shape[0]

    np.random.seed(0)
    f_train= f(X) + 0.01 * np.random.randn(N)
    dfdx_train= df_dx(X) + 0.01 * np.random.randn(N)
    dfdy_train= df_dy(X) + 0.01 * np.random.randn(N)
    ddfdxdy_train= ddf_dxdy(X) + 0.01 * np.random.randn(N)
    
    return np.hstack([f_train[:, None], dfdx_train[:, None], dfdy_train[:, None], ddfdxdy_train[:, None]])

f = lambda x: np.sin(x[:, 0]) +x[:, 1]**3

df_dx = lambda x: np.cos(x[:, 0]) 
df_dy = lambda x: 3.0 * x[:, 1] ** 2
ddf_dxdy = lambda x: x[:, 0]*0.0 

X_grid = create_spatial_grid(-10.0, 10.0, -1.0, 1.0, 100, 100)
f_grid = f(X_grid)
df_dx_grid = df_dx(X_grid)
df_dy_grid = df_dy(X_grid)
ddf_dx_dy_grid = ddf_dxdy(X_grid)

X_train = create_spatial_grid(-10.0, 10.0, -1.0, 1.0, 10, 10)
Y_train = get_data(X_train)

# test on all points
Y_test = np.copy(Y_train)

# only keep f in lower left quadrant
Y_train[~((X_train[:, 0] < 0) & (X_train[:, 1] < 0)), 0] = np.NaN

X, Y = X_train, Y_train


Y = np.hstack([Y[:, [0]], Y[:, [2]], Y[:, [1]], Y[:, [3]]])

print('X: ', X.shape)
print('Y: ', Y.shape, np.nanmean(Y, axis=0))

if False:
    # hierarchical on time
    # contruct CVI model equivalent

    # construct model

    base_kernel = ScaleKernel(
        Matern52(
            input_dim = 1, lengthscales = [ls], active_dims=[0]
        )
    , 1.0) * RBF(
        lengthscales=[ls], active_dims=[1], input_dim=1
    )


    lik_arr = [Gaussian(noise) for i in range(4)]

    # construct P(T)

    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel
        ),
        kernel = FirstOrderDerivativeKernel(base_kernel, input_index = 0),
        mean = FirstOrderDerivativeMean(parent_output_dim=1),
        is_base = True,
        has_parent=False,
        hierarchical=False
    )

    # construct P(S | T)
    diff_op_prior = DifferentialOperatorJoint(
        diff_op_prior_time,
        kernel = FirstOrderDerivativeKernel(input_index = 1, parent_output_dim=2),
        mean = FirstOrderDerivativeMean(input_index = 1, parent_output_dim=1),
        is_base = True,
        has_parent=True,
        hierarchical=True
    )


    # only need to learn f
    q = FullGaussianApproximatePosterior(dim = X.shape[0] * diff_op_prior_time.output_dim)

    # Create Model
    m = stgp.models.GP(
        data = stgp.data.Data(X, Y),
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q
    )

elif True:
    # hierarchical on time
    # contruct CVI model equivalent

    # construct model
    Y = np.hstack([
        Y[:, [0]], Y[:, [1]], Y[:, [2]], Y[:, [3]], Y[:, [2]]*np.NaN, Y[:, [2]]*np.NaN
    ])



    base_kernel = ScaleKernel(
        Matern52(
            input_dim = 1, lengthscales = [ls], active_dims=[0]
        )
    , 1.0) * RBF(
        lengthscales=[ls], active_dims=[1], input_dim=1
    )


    lik_arr = [Gaussian(noise) for i in range(6)]

    # construct P(T)

    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel
        ),
        kernel = SecondOrderDerivativeKernel(base_kernel, input_index = 0),
        mean = SecondOrderDerivativeMean(input_index = 0, parent_output_dim=1),
        is_base = True,
        has_parent=False,
        hierarchical=False
    )

    # construct P(S | T)
    diff_op_prior = DifferentialOperatorJoint(
        diff_op_prior_time,
        kernel = FirstOrderDerivativeKernel(input_index = 1, parent_output_dim=3),
        mean = FirstOrderDerivativeMean(input_index = 1, parent_output_dim=1),
        is_base = True,
        has_parent=True,
        hierarchical=True
    )


    # only need to learn f
    q = FullGaussianApproximatePosterior(dim = X.shape[0] * diff_op_prior_time.output_dim)

    # Create Model
    m = stgp.models.GP(
        data = stgp.data.Data(X, Y),
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q
    )

elif False:
    # hierarchical on time
    # contruct CVI model equivalent

    # construct model
    #Y = np.hstack([y[:, None],  dy_x1[:, None], dy_x2[:, None], dy_x1[:, None]*np.NaN])

    data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

    time_kernel = Matern32(
        input_dim = 1, lengthscales = [ls], active_dims=[0]
    )
    space_kernel = ScaleKernel(RBF(
        lengthscales=[ls], active_dims=[1], input_dim=1
    ), 1.0)

    base_kernel = time_kernel * space_kernel

    base_sde_kernel = SpatioTemporalSeperableKernel(
        FirstOrderDerivativeKernel(time_kernel, input_index=0), 
        space_kernel
    )

    lik_arr = [Gaussian(noise) for i in range(4)]

    # construct P(T)

    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z_ref=data._X), 
            kernel = base_kernel
        ),
        kernel = FirstOrderDerivativeKernel(base_kernel, input_index = 0),
        mean = FirstOrderDerivativeMean(parent_output_dim=1),
        is_base = True,
        has_parent=False,
        hierarchical=False
    )

    # construct P(S | T)
    diff_op_prior = DifferentialOperatorJoint(
        diff_op_prior_time,
        kernel = FirstOrderDerivativeKernel(input_index = 1, parent_output_dim=1),
        mean = FirstOrderDerivativeMean(input_index = 1, parent_output_dim=1),
        is_base = True,
        has_parent=True,
        hierarchical=True
    )

    # surrogate model prior
    latent_sde_gp = GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_sde_kernel
    )

    latent_sde_gp = Independent([latent_sde_gp])

    if False:
        latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp)
    else:
        latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(latent_sde_gp, keep_dims=[0, 1])

    Q = diff_op_prior_time.output_dim
    B = data.Ns * Q
    q = FullConjugateGaussian(
        X = data._X,
        num_latents =  Q,
        block_size= B,
        num_blocks = data.Nt,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            # in state-space format
            data = stgp.data.SpatioTemporalData(X=X, Y=np.reshape(Y, [data.Nt,  Q, data.Ns]), sort=False, train_y=True), # we need gradients Y so set to be trainable
            likelihood=likelihood, 
            prior=latent_sde_gp,
            inference='Sequential',
            full_state_observed = True
        )
    )

    # Create Model
    m = stgp.models.GP(
        data = data,
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q
    )

else:
    # hierarchical on time
    # contruct CVI model equivalent

    # construct model
    MASK = False

    if MASK:
        Y = Y


    else:
        Y = np.hstack([
            Y[:, [0]], Y[:, [1]], Y[:, [2]], Y[:, [3]], Y[:, [2]]*np.NaN, Y[:, [2]]*np.NaN
        ])

    data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

    time_kernel = Matern52(
        input_dim = 1, lengthscales = [ls], active_dims=[0]
    )
    space_kernel = ScaleKernel(RBF(
        lengthscales=[ls], active_dims=[1], input_dim=1
    ), 1.0)

    base_kernel = time_kernel * space_kernel

    if MASK:
        sde_base_time_diff_kernel = FirstOrderDerivativeKernel(time_kernel, input_index=0)
        base_time_diff_kernel = FirstOrderDerivativeKernel(base_kernel, input_index=0)
    else:
        sde_base_time_diff_kernel = SecondOrderDerivativeKernel(time_kernel, input_index=0)
        base_time_diff_kernel = SecondOrderDerivativeKernel(base_kernel, input_index=0)

    base_sde_kernel = SpatioTemporalSeperableKernel(
        sde_base_time_diff_kernel, 
        space_kernel
    )

    if MASK:
        lik_arr = [Gaussian(noise) for i in range(4)]
    else:
        lik_arr = [Gaussian(noise) for i in range(6)]

    # construct P(T)

    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z_ref=data._X), 
            kernel = base_kernel
        ),
        kernel = base_time_diff_kernel,
        mean = None, 
        is_base = True,
        has_parent=False,
        hierarchical=False
    )

    # construct P(S | T)
    diff_op_prior = DifferentialOperatorJoint(
        diff_op_prior_time,
        kernel = FirstOrderDerivativeKernel(input_index = 1, parent_output_dim=1), # do not pass actual parent_output_dim here because with the kronecker structruee they are computed separately
        mean = None,
        is_base = True,
        has_parent=True,
        hierarchical=True
    )

    # surrogate model prior
    latent_sde_gp = GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_sde_kernel
    )

    latent_sde_gp = Independent([latent_sde_gp])

    if MASK:
        latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(latent_sde_gp, keep_dims=[0, 1])
    else:
        latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp)

    Q = diff_op_prior_time.output_dim
    B = data.Ns * Q
    q = FullConjugateGaussian(
        X = data._X,
        num_latents =  Q,
        block_size= B,
        num_blocks = data.Nt,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            # in state-space format
            data = stgp.data.SpatioTemporalData(X=X, Y=np.reshape(Y, [data.Nt,  Q, data.Ns]), sort=False, train_y=True), # we need gradients Y so set to be trainable
            likelihood=likelihood, 
            prior=latent_sde_gp,
            inference='Sequential',
            full_state_observed = True
        )
    )

    # Create Model
    m = stgp.models.GP(
        data = data,
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q
    )


m.print()




if True:
    NatGradTrainer(m).train(1.0, 1)
else:
    trainer = VB_NG_ADAM(m)

    max_iter = 1000
    lc_arr, _ = trainer.train([0.01, 1.0], [max_iter, [1, 1]], callback=progress_bar_callback(max_iter))
    plt.plot(lc_arr)
    plt.show()

print('trained')

print('ELBO: ', m.get_objective())
breakpoint()

pred_mu, pred_var = m.predict_f(X_grid)

print(pred_mu.shape, np.sum(pred_mu), np.sum(pred_var))
print(pred_mu.shape, pred_var.shape, np.sum(pred_mu[:, 0]), np.sum(pred_var[:, 0]))

f_im, f_extents = get_im(X_grid, pred_mu[:, 0])


plt.imshow(f_im, extent=f_extents, origin='lower', aspect='auto')
plt.show()





