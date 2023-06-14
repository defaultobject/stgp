import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp

import objax

import numpy as np

import stgp
from stgp import settings
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.trainers.standard import VB_NG_ADAM, ADAM

from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32, SpatioTemporalSeperableKernel

from stgp.means.mean import FirstOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel, FirstOrderDerivativeKernel_2D
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms import Independent
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.data import Data, TemporallyGroupedData
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/ohamelijnck/Documents/projects/stgp/examples')
from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors

stgp.settings.jitter = 1e-7

# construct 2d grid for X
NS = 15
XS, X, _ = single_output_spatial_data(10, 10, NS, NS, seed=0)
N = X.shape[0]

# Construct data
f = lambda x1, x2: 10*np.sin(10*x1*x2)
df_x1 = lambda x1, x2: 100*np.cos(10*x1 * x2)* 10 * x2
df_x2 = lambda x1, x2: 80*np.cos(10*x1 * x2)* 10 * x1

np.random.seed(0)
y = f(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x1 = df_x1(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x2 = df_x2(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01

Y = np.hstack([y[:, None], dy_x2[:, None], dy_x1[:, None], dy_x1[:, None]*np.NaN])
#Y = np.hstack([y[:, None], dy_x1[:, None]])

print('X: ', X.shape)
print('Y: ', Y.shape, np.nanmean(Y, axis=0))

if False:
    # hierarchical on time
    # contruct CVI model equivalent

    # construct model

    base_kernel = ScaleKernel(
        Matern32(
            input_dim = 1, lengthscales = [0.1], active_dims=[0]
        )
    , 1.0) * RBF(
        lengthscales=[0.1], active_dims=[1], input_dim=1
    )



    lik_arr = [Gaussian(0.1), Gaussian(0.1), Gaussian(0.1), Gaussian(0.1)]

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
else:
    # hierarchical on time
    # contruct CVI model equivalent

    # construct model
    #Y = np.hstack([y[:, None],  dy_x1[:, None], dy_x2[:, None], dy_x1[:, None]*np.NaN])
    Y = np.hstack([y[:, None],   dy_x2[:, None], dy_x1[:, None], dy_x1[:, None]*np.NaN])

    data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

    time_kernel = Matern32(
        input_dim = 1, lengthscales = [0.1], active_dims=[0]
    )
    space_kernel = ScaleKernel(RBF(
        lengthscales=[0.1], active_dims=[1], input_dim=1
    ), 1.0)

    base_kernel = time_kernel * space_kernel

    base_sde_kernel = SpatioTemporalSeperableKernel(
        FirstOrderDerivativeKernel(time_kernel, input_index=0), 
        space_kernel
    )

    lik_arr = [Gaussian(0.1), Gaussian(0.1), Gaussian(0.1), Gaussian(0.1)]
    for lik in lik_arr:
        lik.fix()

    if False:
        #Z_s = np.linspace(np.min(data.X_space), np.max(data.X_space), 10)[:, None]
        Z_s = data.X_space
        Z_sparsity = stgp.sparsity.SpatialSparsity(data.X_time, Z_s, train=False)
        Ms = Z_sparsity.raw_Z.Ns
        Nt = data.Nt
    else:
        Z_s = np.linspace(np.min(data.X_space), np.max(data.X_space), 5)[:, None]
        Z_sparsity = stgp.sparsity.SpatialSparsity(data.X_time, Z_s, train=True)
        Ms = Z_sparsity.raw_Z.Ns
        Nt = data.Nt


    # construct P(T)
    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=Z_sparsity, 
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
        sparsity=Z_sparsity, 
        kernel = base_sde_kernel
    )

    latent_sde_gp = Independent([latent_sde_gp])
    latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp)

    Q = diff_op_prior_time.output_dim
    B = Ms * Q
    q = FullConjugateGaussian(
        X = Z_sparsity,
        num_latents =  Q,
        block_size= B,
        num_blocks = data.Nt,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            # in state-space format
            data = stgp.data.SpatioTemporalData(X=X.raw_Z, Y=np.reshape(Y, [data.Nt,  Q, Ms]), sort=False, train_y=True), # we need gradients Y so set to be trainable
            likelihood=likelihood, 
            prior=latent_sde_gp,
            inference='Sequential',
            full_state_observed = True,
            parallel=True
        )
    )

    data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
    #data = TemporallyGroupedData(X, Y)

    # Create Model
    m = stgp.models.GP(
        data = data,
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q
    )

m.print()

jitted_objective = objax.Jit(m.get_objective, m.vars())

print(m.get_objective())
print(jitted_objective())


if False:
    NatGradTrainer(m).train(1.0, 1)
elif True:
    trainer = VB_NG_ADAM(m)
    lc, _ = trainer.train([0.01, 1.0], [100, [1, 1]])
    plt.plot(lc)
    plt.show()
else:

    trainer = ADAM(m)
    lc, _ =  trainer.train(0.01, 1000)
    plt.plot(lc)
    plt.show()

print(jitted_objective())

pred_mu, pred_var = m.predict_f(XS)

print(pred_mu.shape, np.sum(pred_mu), np.sum(pred_var))
print(pred_mu.shape, pred_var.shape, np.sum(pred_mu[:, 0]), np.sum(pred_var[:, 0]))


plt.imshow(pred_mu[:, 0].reshape(NS, NS)); 
plt.show()





