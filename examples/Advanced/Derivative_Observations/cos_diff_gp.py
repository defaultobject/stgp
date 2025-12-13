"""Batch Gaussian Process Regression with a Derivative Observations"""

from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_disable_jit", True)


import numpy as np

import stgp
from stgp.trainers import ScipyTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import ApproxSDEPeriodic
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.likelihood import DiagonalGaussian, ReshapedGaussian
from stgp.models import GP

import matplotlib.pyplot as plt

# Construct data
f = lambda x: np.cos(10 * x) + np.cos(np.pi * 5 * x)
df = lambda x: -np.sin(10 * x) * 10 - np.sin(np.pi * 5 * x) * np.pi * 5

N = 1000
x = np.linspace(-10, 10, N)
y = f(x) + 0.01 * np.random.rand(N)
dy = df(x) + 0.01 * np.random.rand(N)

X = x[:, None]
Y_all = np.hstack([y[:, None], dy[:, None] * np.NaN])

# remove non derivative observations
Y = np.copy(Y_all)
# Y[int(N*0.2):, 0] = np.NaN

# testing locations
XS = np.linspace(-10, 10, 1000)[:, None]

if False:
    fig, axes = plt.subplots(2)

    axes[0].scatter(X[:, 0], Y_all[:, 0], c="grey")
    axes[0].scatter(X[:, 0], Y[:, 0], c="black")

    axes[1].scatter(X[:, 0], Y_all[:, 1], c="grey")
    axes[1].scatter(X[:, 0], Y[:, 1], c="black")

    plt.show()

# construct model kernel and likelihood
# base_kernel_1d = ScaledMatern32(input_dim = 1, lengthscales = [0.1], variance=1.0)
base_kernel_1d = ApproxSDEPeriodic(100.0, 1.0, 1.0, n_terms=6)
latent_gp = GP(sparsity=stgp.sparsity.NoSparsity(Z=X), kernel=base_kernel_1d)
# latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))
latent_gp = LTI_SDE(Independent([latent_gp]))

F, L, Qc, H, Pinf = latent_gp.state_space_representation(None, 1.0, 0.0)
breakpoint()

lik = ReshapedGaussian(DiagonalGaussian([0.1, 0.1]), N, 2)
lik.fix()

# Create Model
m = stgp.models.GP(
    data=stgp.data.MultiOutputTemporalData(X, Y, sort=True),
    prior=latent_gp,
    likelihood=lik,
    full_state_observed=False,
    inference="Sequential",
)

m.print()

# train
if True:
    print(m.get_objective())
    max_iters = 100
    trainer = ScipyTrainer(m, "L-BFGS-B")
    trainer.train(None, max_iters, callback=progress_bar_callback(max_iters))
    print(m.get_objective())
    m.print()
else:
    print(m.get_objective())

# predict
pred_mu, pred_var = m.predict_f(XS)
pred_mu = pred_mu[..., 0]
pred_var = pred_var[:, 0, :, 0]

# plot
D = 2

fig, axes = plt.subplots(2)
for d in range(D):
    axes[d].fill_between(
        XS[:, 0],
        pred_mu[:, d] - 1.96 * np.sqrt(pred_var[:, d]),
        pred_mu[:, d] + 1.96 * np.sqrt(pred_var[:, d]),
        alpha=0.4,
    )

    axes[d].plot(XS, pred_mu[:, d])

    axes[d].scatter(X, Y_all[:, d], c="grey")

    axes[d].scatter(X, Y[:, d], c="black")

axes[1].scatter(XS, df(XS[:, 0]), c="black")
plt.show()
