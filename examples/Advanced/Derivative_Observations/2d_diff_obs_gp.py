from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_disable_jit", False)


import numpy as np

import stgp
from stgp.kernels import RBF, ScaleKernel, Matern32
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms.pdes import DifferentialOperatorJoint

import matplotlib.pyplot as plt

import sys

sys.path.append("/Users/oliverhamelijnck/Documents/projects/stgp/examples")
from example_utils.data_zoo import single_output_spatial_data

stgp.settings.jitter = 1e-4

# construct 2d grid for X
NS = 15
XS, X, _ = single_output_spatial_data(10, 10, NS, NS, seed=0)
N = X.shape[0]

# Construct data
f = lambda x1, x2: np.sin(10 * x1 * x2)
df_x1 = lambda x1, x2: np.cos(10 * x1 * x2) * 10 * x2
df_x2 = lambda x1, x2: np.cos(10 * x1 * x2) * 10 * x1

np.random.seed(0)
y = f(X[:, 0], X[:, 1]) + np.random.randn(N) * 0.01
dy_x1 = df_x1(X[:, 0], X[:, 1]) + np.random.randn(N) * 0.01
dy_x2 = df_x2(X[:, 0], X[:, 1]) + np.random.randn(N) * 0.01

Y = np.hstack([y[:, None], dy_x2[:, None], dy_x1[:, None], dy_x1[:, None] * np.NaN])
# Y = np.hstack([y[:, None], dy_x1[:, None]])

print("X: ", X.shape)
print("Y: ", Y.shape, np.nanmean(Y, axis=0))

# construct model

base_kernel = ScaleKernel(
    Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]), 1.0
) * RBF(lengthscales=[0.1], active_dims=[1], input_dim=1)

if False:
    kern = FirstOrderDerivativeKernel(base_kernel, input_index=0)
    lik_arr = [Gaussian(0.1), Gaussian(0.1)]

else:
    kern = FirstOrderDerivativeKernel(
        FirstOrderDerivativeKernel(base_kernel, input_index=0),
        input_index=1,
        parent_output_dim=2,
    )
    lik_arr = [Gaussian(0.1), Gaussian(0.1), Gaussian(0.1), Gaussian(0.1)]

diff_op_prior = DifferentialOperatorJoint(
    GP(sparsity=stgp.sparsity.NoSparsity(Z=X), kernel=base_kernel),
    kernel=kern,
    is_base=True,
    has_parent=False,
)

# Create Model
m = stgp.models.GP(
    data=stgp.data.Data(X, Y),
    prior=diff_op_prior,
    likelihood=lik_arr,
)

print(m.get_objective())

pred_mu, pred_var = m.predict_f(XS)

print(pred_mu.shape, np.sum(pred_mu), np.sum(pred_var))
print(pred_mu.shape, pred_var.shape, np.sum(pred_mu[:, 0]), np.sum(pred_var[:, 0]))

plt.imshow(pred_mu[:, 0].reshape(NS, NS))
plt.show()
