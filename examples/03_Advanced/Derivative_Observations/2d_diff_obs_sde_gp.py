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
from stgp.likelihood import Gaussian, BlockDiagonalGaussian
from stgp.models import GP
from stgp.transforms import Independent
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs
from stgp.data import Data

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/ohamelijnck/Documents/projects/stgp/examples')
from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors

# construct 2d grid for X
XS, X, _ = single_output_spatial_data(20, 20, 200, 200, seed=0)
N = X.shape[0]

# Construct data
f = lambda x1, x2: np.sin(10*x1*x2)
df_x1 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x2
df_x2 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x1

y = f(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x1 = df_x1(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x2 = df_x2(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01

Y = np.hstack([y[:, None], dy_x1[:, None], dy_x1[:, None], dy_x1[:, None] * np.NaN])

print('X: ', X.shape)
print('Y: ', Y.shape)

# construct model

data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

base_kernel = SpatioTemporalSeperableKernel(
    Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
    RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])
)

latent_gp = GP(
    sparsity=stgp.sparsity.NoSparsity(Z=X), 
    kernel = base_kernel
)
latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))

Q = 4
var = 0.1 * np.tile(np.eye(Q * data.Ns), [data.Nt, 1, 1]) 
# block diagonal likelihood
lik = BlockDiagonalGaussian(
    block_size = Q * data.Ns,
    num_blocks = data.Nt,
    num_latents = Q,
    variance = var
)
lik.fix()

# Create Model
m = stgp.models.GP(
    data = data,
    prior = latent_gp,
    likelihood = lik,
    inference='Sequential'
)

print(m.get_objective())

pred_mu, pred_var = m.predict_f(XS)
print(pred_mu)
breakpoint()


