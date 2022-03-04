import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp
import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.approximate_posteriors import MeanFieldApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior , MeanFieldConjugateGaussian

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

XS = np.linspace(-1, 2, 1000)[:, None]
x = np.linspace(0, 1, N)
y1 = np.sin(x*10)+0.01*np.random.randn(N)

X = x[:, None]
Y1 = y1[:, None]

Y = np.hstack([Y1 for p in range(P)])

assert Y.shape[1] == P

K = lego.kernels.ScaleKernel(lego.kernels.RBF(lengthscales=[0.1]))

cvi_q = MeanFieldConjugateGaussian([
    legogp.approximate_posteriors.ConjugateGaussian(
        X=X,
        surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, kernel=K, likelihood=likelihood) # batch gp surrogate model
    )
])

m = lego.models.GP(
    X,
    Y,
    inference='Variational',
    whiten=False,
    minibatch_size=None,
    kernel = K,
    likelihood = [lego.likelihood.Gaussian(0.1)],
    approximate_posterior = cvi_q
)
print(m.get_objective())
breakpoint()

natgrad_trainer = NatGradTrainer(m, schedule=None)
natgrad_trainer.train(1.0, 1)
#breakpoint()


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
