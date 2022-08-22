""" Single GP regression """
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)


import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import matplotlib.pyplot as plt
from pathlib import Path

import gpflow
from gpflow.models import VGP, GPR, SGPR, SVGP
from gpflow.optimizers import NaturalGradient
import tensorflow as tf
from gpflow.optimizers.natgrad import XiSqrtMeanVar

from tqdm import trange

# Fix randomness
np.random.seed(0)

# Generate Data
N = 100

x = np.linspace(0, 1, N)
y = np.sin(x*10) + 0.1*np.random.randn(N)
X = x[:, None]
Y = y[:, None]

XS = np.linspace(-1, 2, 1000)[:, None]

inducing_variable = X

m_gpflow = SVGP(
    kernel = gpflow.kernels.RBF(lengthscales=0.1, variance=1.0),
    likelihood = gpflow.likelihoods.Gaussian(variance=0.1),
    inducing_variable = inducing_variable,
    whiten = False
)

epochs = 1000
lr = 0.01

lc_arr = []
data = (X, Y)
adam_optimizer = tf.optimizers.Adam(lr)
svgp_natgrad_loss = m_gpflow.training_loss_closure(data)
for i in trange(epochs):
    adam_optimizer.minimize(svgp_natgrad_loss, m_gpflow.trainable_variables)
    lc_arr.append(
        -m_gpflow.elbo(data).numpy()
    )

plt.plot(lc_arr)
plt.show()

# Predict
pred_mu, pred_var = m_gpflow.predict_y(XS)

pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)

# Plot results
fig = plt.figure(figsize=(10, 5))
ax = plt.gca()

ax.fill_between(np.squeeze(XS), np.squeeze(pred_mu - 2*np.sqrt(pred_var)), np.squeeze(pred_mu + 2*np.sqrt(pred_var)), alpha=0.4)
ax.plot(XS, pred_mu)
ax.scatter(X, Y)
plt.show()

