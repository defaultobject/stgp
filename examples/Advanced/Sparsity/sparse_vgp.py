"""Sparse Variational Gaussian Process Regression"""

import sys

sys.path.append("../../")

from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_disable_jit", False)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data
from stgp.transforms import Independent
from stgp.trainers.standard import VB_NG_ADAM

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

stgp.settings.jitter = 1e-7
stgp.settings.ng_jitter = 1e-8

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

# Construct Model
# setup 10 inducing points
Z = np.linspace(np.min(X), np.max(X), 10)[:, None]

data = Data(X, Y, minibatch_size=10, seed=0)
# data = Data(X, Y)

m = GP(
    data=data,
    prior=Independent(
        [
            GP(
                sparsity=stgp.sparsity.FullSparsity(Z=Z),
                kernel=ScaleKernel(RBF(input_dim=1, lengthscales=[0.1]), 1.0),
                prior=True,
            )
        ]
    ),
    likelihood=ProductLikelihood([Gaussian(0.1)]),
    inference="Variational",
)

# print(m.get_objective())
# exit()

obj = objax.Jit(m.get_objective, m.vars())
if False:
    while True:
        data.batch()
        plt.hist(data.idx)
        plt.show()
        breakpoint()

    indexes = []
    for i in range(100):
        data.batch()
        indexes.append(np.unique(data.idx).shape[0])

    evals = [obj() for i in range(1000)]

    breakpoint()

    print(np.mean(evals))

    plt.hist(evals)
    plt.show()
    exit()


# Train
if False:
    # ng_trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton')
    ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()

    ng_trainer.train(1.0, 1)
    print(m.get_objective())
    print(m.get_objective())
    breakpoint()

# trainer = VB_NG_ADAM(m, enforce_psd_type='laplace_gauss_newton')
trainer = VB_NG_ADAM(m)

max_iters = 100
lc_arr, _ = trainer.train(
    [0.01, 0.99], [max_iters, [1, 1]], callback=progress_bar_callback(max_iters)
)

if False:
    evals = [obj() for i in range(1000)]

    plt.hist(evals)
    plt.show()

    breakpoint()

plt.plot(lc_arr[::2])
plt.show()

pred_mu, pred_var = m.predict_y(XS)

plt.fill_between(
    np.squeeze(XS),
    np.squeeze(pred_mu - 1.96 * np.sqrt(pred_var)),
    np.squeeze(pred_mu + 1.96 * np.sqrt(pred_var)),
    facecolor=colors.LINE_COL,
    alpha=0.3,
)
plt.plot(XS, pred_mu, color=colors.LINE_COL, label="GP Fit")
plt.scatter(X, Y, color="grey", label="Training Data")

# plot inducing locations and values
plt.scatter(
    m.prior.get_sparsity_list()[0].Z,
    np.zeros_like(np.squeeze(m.approximate_posterior.m[0])),
    0.5,
    c="black",
)

plt.scatter(
    m.prior.get_sparsity_list()[0].Z,
    np.squeeze(m.approximate_posterior.m[0]),
    c="black",
    label="Inducing Points",
)

plt.legend()
plt.show()
