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
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms.pdes import DifferentialOperatorJoint
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

base_kernel = ScaleKernel(Matern32(input_dim = 2, lengthscales = [0.1, 0.1]), 1.0)

kern = FirstOrderDerivativeKernel(
    FirstOrderDerivativeKernel(base_kernel, input_index = 0),
    input_index = 0,
    parent_output_dim = 2
)

diff_op_prior = DifferentialOperatorJoint(
    GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_kernel
    ),
    kernel = kern,
    is_base = True,
    has_parent=False
)

# Create Model
m = stgp.models.GP(
    data = stgp.data.Data(X, Y),
    prior = diff_op_prior,
    likelihood = [Gaussian(0.1), Gaussian(0.1), Gaussian(0.1), Gaussian(0.1)],
)

print(m.get_objective())

pred_mu, pred_var = m.predict_f(XS)
print(pred_mu)
breakpoint()

