import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.computation.natural_gradients.nat_grad import general_ell_natural_gradients
from legogp.approximate_posteriors import MeanFieldApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior 

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
M = 30

XS = np.linspace(-2, 3, 1000)[:, None]
x = np.linspace(0, 1, N)
y1 = np.sin(x*10)+0.01*np.random.randn(N)

X = x[:, None]
Y1 = y1[:, None]

Y = np.hstack([Y1 for p in range(P)])+2.0

assert Y.shape[1] == P

K1 = lego.kernels.deep_kernels.DeepRBF()


qu = lego.approximate_posteriors.MeanFieldApproximatePosterior(
    approximate_posteriors = [MM_GaussianInnerLayerApproximatePosterior(kernel=K1, dim=M)]
)

Z = np.linspace(0, 1, M)[:, None]

m = lego.models.GP(
    X,
    Y,
    Z = Z,
    inference='Variational',
    whiten=False,
    minibatch_size=100,
    kernel = lego.kernels.ScaleKernel(lego.kernels.RBF(lengthscales=[0.1])),
    likelihood = lego.likelihood.Gaussian(0.1),
    approximate_posterior = qu
)

epochs = 5
callback = progress_bar_callback(epochs)
learning_curve, training_time = NatGradTrainer().train(
    m, 
    None,
    1.0,
    epochs,
    callback = callback
)

breakpoint()


m.get_objective()

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
