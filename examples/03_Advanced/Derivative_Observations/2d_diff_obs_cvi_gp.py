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
from stgp.likelihood import Gaussian, BlockDiagonalGaussian, PrecisionBlockDiagonalGaussian
from stgp.models import GP
from stgp.transforms import Independent
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE
from stgp.data import Data
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior, FullConjugatePrecisionGaussian

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/ohamelijnck/Documents/projects/stgp/examples')
from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors

stgp.settings.jitter = 1e-4

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

Y = np.hstack([y[:, None], dy_x2[:, None], dy_x1[:, None], dy_x1[:, None]*np.NaN])

print('X: ', X.shape)
print('Y: ', Y.shape, np.nanmean(Y, axis=0))

# there are 4 latents functions, f, df/ds, df/dt, d^2f/(dtds)
Q = 4
data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

base_kernel = ScaleKernel(
    Matern32(
        input_dim = 1, lengthscales = [0.1], active_dims=[0]
    )
, 1.0) * RBF(
    lengthscales=[0.1], active_dims=[1], input_dim=1
)

kern = FirstOrderDerivativeKernel(
    FirstOrderDerivativeKernel(base_kernel, input_index = 0),
    input_index = 1,
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

latent_gp = Independent([diff_op_prior])


lik = [Gaussian(0.1) for q in range(Q)]

# use full gaussian for consistency
B = data.Ns * Q
q = FullConjugateGaussian(
    X = data._X,
    num_latents =  Q,
    block_size= B,
    num_blocks = data.Nt,
    surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
        # in time-state-space format
        data = stgp.data.DataTPS(X, Y=np.reshape(Y, [-1, Q*data.Ns ]), num_latents=Q),
        likelihood=likelihood, 
        prior=latent_gp
    )
)


# Create Model
m = stgp.models.GP(
    data = data,
    prior = latent_gp,
    likelihood = lik,
    inference='Variational',
    approximate_posterior = q
)
print(m.get_objective())

NatGradTrainer(m).train(1.0, 1)

print(m.get_objective())

pred_mu, pred_var = m.predict_f(XS)

print(pred_mu.shape, np.sum(pred_mu), np.sum(pred_var))
print(pred_mu.shape, pred_var.shape, np.sum(pred_mu[:, 0]), np.sum(pred_var[:, 0]))

breakpoint()

plt.imshow(pred_mu[:, 0].reshape(NS, NS)); 
plt.show()


