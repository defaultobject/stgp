""" Single GP regression """
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import RBF, ScaleKernel, BiasKernel
from legogp.likelihood import Gaussian

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

# Fix randomness
np.random.seed(0)

# Generate Data
N = 100

x = np.linspace(0, 1, N)
y = np.sin(x*10) + 0.1*np.random.randn(N)
X = x[:, None]
Y = y[:, None]

XS = np.linspace(-1, 2, 1000)[:, None]

# Create Model
m = lego.models.GP(
    X, 
    Y, 
    kernel=ScaleKernel(RBF(lengthscales=[0.1])),
    likelihood = Gaussian(variance=0.1)
)

print(m.get_objective())

if True:
    # Train
    epochs = 200
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
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
