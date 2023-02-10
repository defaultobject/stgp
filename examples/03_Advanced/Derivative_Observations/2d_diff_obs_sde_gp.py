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
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE
from stgp.data import Data

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
f = lambda x1, x2: np.sin(10*x1*x2)
df_x1 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x2
df_x2 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x1

np.random.seed(0)
y = f(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x1 = df_x1(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
dy_x2 = df_x2(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01

no_diff_flag = False
include_ds = True

# only uses time derivates
if no_diff_flag:
    Y = y[:, None]
else:
    if include_ds:
        #Y = np.hstack([y[:, None],  dy_x1[:, None]*np.NaN, dy_x1[:, None],  dy_x1[:, None]*np.NaN])
        Y = np.hstack([y[:, None], dy_x2[:, None], dy_x1[:, None], dy_x1[:, None]*np.NaN])
    else:
        Y = np.hstack([y[:, None], dy_x1[:, None]])


print('X: ', X.shape)
print('Y: ', Y.shape, np.nanmean(Y, axis=0))

# construct model

data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

if not no_diff_flag:
    if not include_ds:
        base_kernel = SpatioTemporalSeperableKernel(
            FirstOrderDerivativeKernel(Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]), input_index=0), 
            RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])
        )
    else:
        # TODO: what is the new format of f?
        # TODO: fix predictions
        base_kernel = SpatioTemporalSeperableKernel(
            FirstOrderDerivativeKernel(Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]), input_index=0), 
            FirstOrderDerivativeKernel(RBF(input_dim=1, lengthscales=[0.1], active_dims=[1]), input_index=1),
            spatial_output_dim = 2
        )
else:
    base_kernel = SpatioTemporalSeperableKernel(
        Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]), 
        RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])
    )

latent_gp = GP(
    sparsity=stgp.sparsity.NoSparsity(Z=X), 
    kernel = base_kernel
)
if no_diff_flag:
    latent_gp = LTI_SDE(Independent([latent_gp]))
    Q = 1
else:
    latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))
    if include_ds:
        Q = 4
    else:
        Q = 2

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
    inference='Sequential',
    full_state_observed=True
)

pred_mu, pred_var = m.predict_f(data.X)
post_mu, post_var = m.posterior_blocks()

from stgp.computation.permutations import data_order_to_output_order, permute_vec

if True:
    #H = data_order_to_output_order(4, data.Ns)
    #H = data_order_to_output_order(data.Ns, 4).T
    #p_post_mu = jax.vmap(lambda a: H @ a)(post_mu)
    p_post_mu = jax.vmap(lambda a: permute_vec(a, 4))(post_mu)
else:
    p_post_mu = post_mu
p_post_mu = np.reshape(p_post_mu, [-1, 4])[..., None]

print(p_post_mu-pred_mu)
print(np.sum(pred_mu) - np.sum(post_mu))
breakpoint()

print(m.get_objective())

pred_mu, pred_var = m.predict_f(XS)

print(pred_mu.shape, np.sum(pred_mu), np.sum(pred_var))
print(pred_mu.shape, pred_var.shape, np.sum(pred_mu[:, 0]), np.sum(pred_var[:, 0]))

breakpoint()

plt.imshow(pred_mu[:, 0].reshape(NS, NS)); 
plt.show()


