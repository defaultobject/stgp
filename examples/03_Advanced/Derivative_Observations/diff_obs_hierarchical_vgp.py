""" Structured Variational Gaussian Process Regression with a Derivative Observations"""

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
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52
from stgp.means.mean import FirstOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms import Independent
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.data import Data
from stgp.trainers.standard import VB_NG_ADAM

import matplotlib.pyplot as plt

# Construct data
f = lambda x: np.sin(10*x)
df = lambda x: np.cos(10*x)*10

N = 20
x = np.linspace(0, 1, N)
y = f(x) + 0.01*np.random.rand(N)
dy = df(x) + 0.01*np.random.rand(N)

X = x[:, None]
Y_all = np.hstack([y[:, None], dy[:, None]])

# remove non derivative observations
Y = np.copy(Y_all)
Y[int(N*0.2):, 0] = np.NaN

# testing locations
XS = np.linspace(-5, 6, 1000)[:, None]

if False:
    fig, axes = plt.subplots(2)

    axes[0].scatter(X[:, 0], Y_all[:, 0], c='grey')
    axes[0].scatter(X[:, 0], Y[:, 0], c='black')

    axes[1].scatter(X[:, 0], Y_all[:, 1], c='grey')
    axes[1].scatter(X[:, 0], Y[:, 1], c='black')

    plt.show()

# construct model

Z = np.linspace(0, 1, 5)[:, None]
#sparsity=stgp.sparsity.FullSparsity(Z=Z)
sparsity=stgp.sparsity.NoSparsity(Z=X)
#base_kernel_1d = ScaleKernel(Matern32(input_dim = 1, lengthscales = [0.1]), 1.0)
base_kernel_1d = ScaleKernel(RBF(input_dim = 1, lengthscales = [0.1]), 1.0)

base_gp = Independent([
    GP(
        sparsity=sparsity, 
        kernel = base_kernel_1d
    )
])

kern = FirstOrderDerivativeKernel(base_kernel_1d, parent_output_dim=base_gp.output_dim)
mean = FirstOrderDerivativeMean(parent_output_dim=base_gp.output_dim)

prior = DifferentialOperatorJoint(
    base_gp,
    mean=mean,
    kernel = kern,
    is_base = True,
    has_parent=True,
    hierarchical = True
)

q = FullGaussianApproximatePosterior(dim = sparsity.Z.shape[0] * base_gp.output_dim)

# Create Model
m = stgp.models.GP(
    data = stgp.data.Data(X, Y),
    prior = prior,
    likelihood = [Gaussian(0.1), Gaussian(0.1)],
    inference='Variational',
    approximate_posterior = q
)

# train
m.print()

if True:
    trainer = VB_NG_ADAM(m)
    lc_arr, _ = trainer.train([1e-3, 1.0], [100, [1, 1]], callback=progress_bar_callback(100))

    print(m.get_objective())

    plt.plot(lc_arr)
    plt.show()

    m.print()
else:
    print(m.get_objective())

# predict
pred_mu, pred_var = m.predict_f(XS)
pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)


# plot
D = 2

fig, axes = plt.subplots(2)
for d in range(D):
    axes[d].fill_between(
        XS[:, 0], 
        pred_mu[:, d] - 1.96*np.sqrt(pred_var[:, d]),
        pred_mu[:, d] + 1.96*np.sqrt(pred_var[:, d]),
        alpha = 0.4
    )

    axes[d].plot(
        XS, pred_mu[:, d]
    )

    axes[d].scatter(
        X, Y_all[:, d], c='grey'
    )

    axes[d].scatter(
        X, Y[:, d], c='black'
    )

plt.show()
