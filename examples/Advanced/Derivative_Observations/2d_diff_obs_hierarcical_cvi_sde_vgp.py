from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_disable_jit", False)


import numpy as np

import stgp
from stgp.trainers import NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.trainers.standard import VB_NG_ADAM, NatGradTrainer

from stgp.kernels import RBF, ScaleKernel, Matern32


from stgp.zoo.sde_diff import diff_cvi_sde_vgp


import matplotlib.pyplot as plt

import sys

sys.path.append("/Users/ohamelijnck/Documents/projects/stgp/examples")
from example_utils.data_zoo import single_output_spatial_data

stgp.settings.jitter = 1e-7

ls = 0.1
noise = 0.1

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
print("X: ", X.shape)
print("Y: ", Y.shape, np.nanmean(Y, axis=0))

time_kernel = Matern32(input_dim=1, lengthscales=[ls], active_dims=[0])
space_kernel = ScaleKernel(RBF(lengthscales=[ls], active_dims=[1], input_dim=1), 1.0)

m = diff_cvi_sde_vgp(
    X,
    Y,
    num_latents=1,
    time_diff=1,
    space_diff=1,
    time_kernel=time_kernel,
    space_kernel=space_kernel,
    lik_var=noise,
    fix_y=True,
    Zs=None,
    keep_dims=[0, 1],
    hierarchical=True,
)


print("training")
if True:
    NatGradTrainer(m).train(1.0, 1)
else:
    trainer = VB_NG_ADAM(m)

    max_iter = 1000
    lc_arr, _ = trainer.train(
        [0.01, 1.0], [max_iter, [1, 1]], callback=progress_bar_callback(max_iter)
    )
    plt.plot(lc_arr)
    plt.show()

print("trained")

print("ELBO: ", m.get_objective())

pred_mu, pred_var = m.predict_f(XS)

print(pred_mu.shape, np.sum(pred_mu), np.sum(pred_var))
print(pred_mu.shape, pred_var.shape, np.sum(pred_mu[:, 0]), np.sum(pred_var[:, 0]))

plt.imshow(pred_mu[:, 0].reshape(NS, NS))

plt.show()
