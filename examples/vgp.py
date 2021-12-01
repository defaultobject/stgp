import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback

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

checkpoint_folder = Path('checkpoints')
checkpoint_folder.mkdir(exist_ok=True)

# generate data
P = 1

N = 100
M = 20

XS = np.linspace(-0.5, 1.5, 1000)[:, None]
x = np.linspace(0, 1, N)
y1 = np.sin(x*10)+0.01*np.random.randn(N)

X = x[:, None]
Y1 = y1[:, None]

Y = np.hstack([Y1 for p in range(P)])

assert Y.shape[1] == P

qu = lego.approximate_posteriors.MeanFieldApproximatePosterior(dim_list=[M])

Z = np.linspace(0, 1, M)[:, None]

m = lego.models.GP(
    X,
    Y,
    Z = Z,
    inference='Variational',
    whiten=False,
    minibatch_size=10,
    kernel = lego.kernels.RBF(lengthscales=[0.1]),
    likelihood = lego.likelihood.Gaussian(0.01),
    approximate_posterior = qu
)

m.get_objective()

restore = False

if restore:
    m.load_from_checkpoint(str(checkpoint_folder / 'vgp'))
else:
    epochs = 500
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )

    plt.plot(learning_curve)
    plt.show()

    print(learning_curve[0], learning_curve[-1])

    m.checkpoint(str(checkpoint_folder / 'vgp'))

pred_mu, pred_var = m.predict_f(XS, squeeze=True)

fig = plt.figure()
plt.fill_between(
    np.squeeze(XS),
    np.squeeze(pred_mu) + 2*np.squeeze(np.sqrt(pred_var)),
    np.squeeze(pred_mu) - 2*np.squeeze(np.sqrt(pred_var)),
    alpha = 0.4
)
plt.plot(XS, pred_mu)
plt.scatter(X, Y, c='black')
plt.show()
