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
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32, SpatioTemporalSeperableKernel
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.likelihood import Gaussian, BlockDiagonalGaussian, ProductLikelihood
from stgp.models import GP
from stgp.transforms import Independent, OutputMap, MultiOutput
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE
from stgp.data import Data
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/ohamelijnck/Documents/projects/stgp/examples')
from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors

stgp.settings.jitter = 1e-4
stgp.settings.verbose = True
stgp.settings.cvi_ng_exploit_space_time = True

# construct 2d grid for X
NS = 15
XS, X, _ = single_output_spatial_data(10, 10, NS, NS, seed=0)
N = X.shape[0]

# Construct data
f = lambda x1, x2: np.sin(10*x1*x2)
df_x1 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x2
df_x2 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x1

np.random.seed(0)
y = f(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x1 = df_x1(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x2 = df_x2(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01

no_diff_flag = False
include_ds = True
whiten_space = False
stgp.settings.whiten_space = False

Y = np.hstack([y[:, None], dy_x2[:, None], dy_x1[:, None], dy_x1[:, None]*np.NaN])


print('X: ', X.shape)
print('Y: ', Y.shape, np.nanmean(Y, axis=0))

# there are 4 latents functions, f, df/ds, df/dt, d^2f/(dtds)
Q = 4
data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

time_kernel = Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0])
spatial_kernel = RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])

# GP prior kernels
base_kernel = SpatioTemporalSeperableKernel(
    time_kernel, 
    spatial_kernel,
    whiten_space = whiten_space
)

# surrogare model kernels using same base a gp prior
base_sde_kernel = SpatioTemporalSeperableKernel(
    FirstOrderDerivativeKernel(time_kernel, input_index=0), 
    FirstOrderDerivativeKernel(spatial_kernel, input_index=1),
    spatial_output_dim = 2,
    whiten_space = whiten_space
)


# surrogate model prior
latent_sde_gp = GP(
    sparsity=stgp.sparsity.NoSparsity(Z=X), 
    kernel = base_sde_kernel
)

latent_sde_gp = Independent([latent_sde_gp])
latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp, whiten_space = whiten_space)

# gp model prior
latent_diff_op = DifferentialOperatorJoint(
    GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_kernel
    ),
    kernel = FirstOrderDerivativeKernel(FirstOrderDerivativeKernel(base_kernel, input_index=0), input_index=1, parent_output_dim = 2),
    is_base = True,
    has_parent = False,
    whiten_space = whiten_space
)


# use full gaussian for consistency
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

prior_outputs = OutputMap(
    latent_diff_op,
    [
        [0], 
        [1], 
        [2]
    ]
)


prior = MultiOutput(prior_outputs)

Y = np.hstack([Y[:, [0]], Y[:, [1]], Y[:, [2]]])

#print('X: ', X.shape)
#print('Y: ', Y.shape, np.nanmean(Y, axis=0))

# there are 3 outputs f, df/ds, df/dt
Q = Y.shape[1]
data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

lik = [ProductLikelihood([Gaussian(0.1)]) for q in range(Q)]


# Create Model
m = stgp.models.GP(
    data = data,
    prior = prior,
    likelihood = lik,
    inference='Variational',
    approximate_posterior = q
)
print(m.get_objective())

NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton_delta_u_mc_f').train(1.0, 1)

print(m.get_objective())

pred_mu, pred_var = m.predict_f(XS)
pred_mu = np.array(pred_mu)
pred_var = np.array(pred_var)

print(pred_mu.shape, np.sum(pred_mu), np.sum(pred_var))
print(pred_mu.shape, pred_var.shape, np.sum(pred_mu[0]), np.sum(pred_var[0]))

plt.imshow(pred_mu[0].reshape(NS, NS)); 
plt.show()


