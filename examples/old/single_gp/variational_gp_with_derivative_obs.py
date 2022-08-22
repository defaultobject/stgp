"""
We mode data generated as
    y = sin( 10 * x) +eps
    dy/dx = 10 cos( 10 x) + e
"""

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp import settings
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel
from stgp.likelihood import Gaussian
from stgp.kernels.diff_op import FirstOrderDerivativeKernel_1D
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.transforms import OutputMap, MultiOutput
from stgp.models import GP
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior

import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import matplotlib.pyplot as plt
from pathlib import Path

# Fix randomness
np.random.seed(0)

# Generate Data
N = 50

x = np.linspace(0, 2, N)

y = np.sin(10 * x) + 0.1*np.random.randn(N)
yt = 10 * np.cos(10*x) + np.random.randn(N)

X = x[:, None]
Y = y[:, None]
Y[25:, :] = np.NaN
Yt = yt[:, None]

XS = np.linspace(-1, 3, 1000)[:, None]

if False:
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].scatter(x, y)
    axes[1].scatter(x, yt)
    plt.show()

# construct model
Y_stacked = np.hstack([Y, Yt])
print(Y_stacked.shape)

data = stgp.data.Data(X, Y_stacked)

base_kernel_1d = RBF(input_dim = 1, lengthscales = [0.1])

diff_op_prior = DifferentialOperatorJoint(
    GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_kernel_1d
    ),
    FirstOrderDerivativeKernel_1D(base_kernel_1d)
)

if  True:
    prior_output_1, prior_output_2 = OutputMap(
        diff_op_prior, 
        [[0], [1]], 
    )

    prior = MultiOutput([
        prior_output_1,
        prior_output_2
    ])
else:
    prior = diff_op_prior


# Defined in data-latent format?
q = FullGaussianApproximatePosterior(dim = N * diff_op_prior.output_dim)


lik_arr = [Gaussian(variance=0.01), Gaussian(variance=0.1)]
lik_arr[0].variance_param.fix()
lik_arr[1].variance_param.fix()

# Create Model
m = stgp.models.GP(
    data = data, 
    prior = prior,
    likelihood = lik_arr,
    inference='Variational',
    approximate_posterior = q,
    prediction_samples = 1000,
    whiten=True
)

print(m.get_objective())

if False:
    ng_trainer = NatGradTrainer(m)
    ng_trainer.train(1.0, 1) 

print(m.get_objective())

if True:
    # Train
    epochs = 100

    callback = progress_bar_callback(epochs)

    learning_curve, training_time = GradDescentTrainer(
        m, 
        objax.optimizer.Adam,
    ).train(
        0.01,
        epochs,
        callback = callback
    )

    # Plot learning curve
    plt.plot(learning_curve)
    plt.show()

print(m.get_objective())

# Predict
pred_mu, pred_var = m.predict_y(XS, squeeze=True)

if True:
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].fill_between(np.squeeze(XS), np.squeeze(pred_mu[0] - 2*np.sqrt(pred_var[0])), np.squeeze(pred_mu[0] + 2*np.sqrt(pred_var[0])), alpha=0.4)
    axes[0].plot(XS, pred_mu[0])
    axes[0].scatter(x, y)

    axes[1].fill_between(np.squeeze(XS), np.squeeze(pred_mu[1] - 2*np.sqrt(pred_var[1])), np.squeeze(pred_mu[1] + 2*np.sqrt(pred_var[1])), alpha=0.4)
    axes[1].plot(XS, pred_mu[1])
    axes[1].scatter(x, yt)
    plt.show()

