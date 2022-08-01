""" Single GP regression """
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel
from stgp.likelihood import Gaussian

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
N = 100

x = np.linspace(0, 1, N)
y = np.sin(x*10) + 0.1*np.random.randn(N)
X = x[:, None]
Y = y[:, None]

XS = np.linspace(-1, 2, 1000)[:, None]

data = stgp.data.Data(X, Y)

# Create Model
m = stgp.models.GP(
    data = data, 
    kernel=ScaleKernel(RBF(lengthscales=[0.1])),
    likelihood = [Gaussian(variance=0.1)],
    inference='Variational',
    whiten = False
)

print(m.get_objective())

if False:
    ng_trainer = NatGradTrainer(m)
    ng_trainer.train(1.0, 1) 

if True:
    m.print()

    # Train
    epochs = 200

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

    m.print()

print(m.get_objective())

# Predict
pred_mu, pred_var = m.predict_y(XS, squeeze=True)

# Plot results
fig = plt.figure(figsize=(10, 5))
ax = plt.gca()

ax.fill_between(np.squeeze(XS), np.squeeze(pred_mu - 2*np.sqrt(pred_var)), np.squeeze(pred_mu + 2*np.sqrt(pred_var)), alpha=0.4)
ax.plot(XS, pred_mu)
ax.scatter(X, Y)
plt.show()
