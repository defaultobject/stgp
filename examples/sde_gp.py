""" Single GP regression computed through Kalman Filtering and Smoothing"""
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp 
from stgp.trainers import GradDescentTrainer, ScipyTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.data import TemporalData
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent
from stgp.models import GP

import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import stdata as st
from stdata.plots import grid_to_matrix
import matplotlib.pyplot as plt
from pathlib import Path

from data_zoo import single_output_timeseries

# Fix randomness
np.random.seed(0)

XS, X, Y = single_output_timeseries(100, 1000, seed=0)

data = TemporalData(X=X, Y=Y, sort=True)

lik = ReshapedGaussian(Gaussian(), num_blocks=data.Nt, block_size=1)

Z = stgp.sparsity.NoSparsity(Z_ref = data.X)
D = 1

latent_gp = GP(
    sparsity = Z, 
    kernel = Matern32(input_dim=D, lengthscales=[0.1 for d in range(D)]),
    prior = True
)

# Create Model
m = GP(
    prior = LTI_SDE(Independent([latent_gp])),
    data = data, 
    likelihood = lik,
    inference='Sequential'
)


print(m.get_objective())

if True:
    # Train
    epochs = 500
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = GradDescentTrainer(
        m, 
        objax.optimizer.Adam
    ).train(
        0.01,
        epochs,
        callback = callback
    )

    # Plot learning curve
    plt.plot(learning_curve)
    plt.show()

# Predict
pred_mu, pred_var = m.predict_y(XS, squeeze=True)

# Plot results
fig = plt.figure(figsize=(10, 5))
ax = plt.gca()

ax.fill_between(np.squeeze(XS), np.squeeze(pred_mu - 2*np.sqrt(pred_var)), np.squeeze(pred_mu + 2*np.sqrt(pred_var)), alpha=0.4)
ax.plot(XS, pred_mu)
ax.scatter(X, Y)
plt.show()

