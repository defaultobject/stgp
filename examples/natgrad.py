import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.computation.natural_gradients.nat_grad import general_ell_natural_gradients

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

qu = lego.approximate_posteriors.MeanFieldApproximatePosterior(dim_list=[M])

Z = np.linspace(0, 1, M)[:, None]

m = lego.models.GP(
    X,
    Y,
    Z = Z,
    inference='Variational',
    whiten=False,
    minibatch_size=100,
    kernel = lego.kernels.ScaleKernel(lego.kernels.RBF(lengthscales=[0.5])) + lego.kernels.ScaleKernel(lego.kernels.BiasKernel()),
    likelihood = lego.likelihood.Gaussian(0.1),
    approximate_posterior = qu
)

general_ell_natural_gradients(m, beta=0.1)


m.get_objective()

