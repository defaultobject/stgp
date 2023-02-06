""" Sparse Variational Gaussian Process Regression """
import sys
sys.path.append('../../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors
from stgp.trainers.standard import VB_NG_ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF 
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data
from stgp.transforms import Independent
from stgp import settings
from tqdm import trange

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

stgp.settings.jitter = 1e-5
# Construct Data
NS_nt = 100
NS_ns = 100
XS, X, Y = single_output_spatial_data(10, 10, NS_nt, NS_ns, seed=0)

# Construct Model
# setup 10 inducing points
M = 3
Z_spatial = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), M)[:, None]

data_st = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
X_time = data_st.X_time

kernel = SpatioTemporalSeperableKernel(
    Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
    RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])
)

data = Data(X, Y)
m = GP(
    data = data,
    prior = Independent([
        GP(
            sparsity = stgp.sparsity.SpatialSparsity(X_time = X_time, Z_space = Z_spatial), 
            kernel = kernel,
            prior = True
        )
    ]),
    likelihood = ProductLikelihood([Gaussian(0.1)]),
    inference='Variational'
)


# Train
if True:
    print(m.get_objective())
    trainer = VB_NG_ADAM(m, enforce_psd_type='laplace_gauss_newton')
    trainer.ng_trainer.train(1.0, 1)
    print(m.get_objective())
    m.print()

    max_iters = 100
    lc, _ = trainer.train([0.01, 1.0], [max_iters, [1, 1]], callback=progress_bar_callback(max_iters))

    plt.plot(lc)
    plt.show()
    print(m.get_objective())

pred_mu, pred_var = m.predict_y(XS)

fig, axes = plt.subplots(1, 2, squeeze=False)
Nt = data_st.Nt
Ns = data_st.Ns

norm = Normalize(np.min(Y[:, 0]), np.max(Y[:, 0]))

axes[0][0].imshow(
    pred_mu.reshape(NS_nt, NS_ns), 
    extent=[np.min(XS[:, 0]), np.max(XS[:, 0]), np.min(XS[:, 1]), np.max(XS[:, 1])],
    origin='lower',
    norm=norm
)
axes[0][0].scatter(X[:, 1], X[:, 0], c=Y[:, 0], edgecolor='white', norm=norm)
plt.show()

breakpoint()


