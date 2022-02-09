import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import RBF, ScaleKernel, BiasKernel

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
P = 10

N = 50

XS1 = np.linspace(-0.5, 1.5, 20)[:, None]
XS2 = np.linspace(0, 1.5, 20)[:, None]
XS_stacked = np.vstack([XS1, XS2])


XS = np.linspace(-0.5, 1.5, 1000)[:, None]
x = np.linspace(0, 1, N)
y1 = np.sin(x*10)+0.01*np.random.randn(N)+2.0
y2 = -np.sin(x*8)+0.01*np.random.randn(N)-1.0

X = x[:, None]
Y1 = y1[:, None]
Y2 = y2[:, None]


m1 = lego.models.GP(X, Y1, kernel=ScaleKernel(RBF(lengthscales=[0.1]))+ScaleKernel(BiasKernel()))
m2 = lego.models.GP(X, Y2, kernel=lego.kernels.deep_kernels.DeepLinear(m1))

model_list = [m2, m1]

epochs = 100

restore = True

if restore:
    m2.load_from_checkpoint(str(checkpoint_folder / 'mf'))
else:
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        model_list, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )
    m2.checkpoint(str(checkpoint_folder / 'mf'))

    plt.plot(learning_curve)
    plt.show()

mu1, var1 = m1.predict_y(XS)
mu2, var2 = m2.predict_y(XS)

fig, axes = plt.subplots(2, 1)

axes[0].fill_between(np.squeeze(XS), np.squeeze(mu1 - 2*np.sqrt(var1)), np.squeeze(mu1 + 2*np.sqrt(var1)), alpha=0.4)
axes[0].plot(XS, mu1)
axes[0].scatter(X, Y1)

axes[1].fill_between(np.squeeze(XS), np.squeeze(mu2 - 2*np.sqrt(var2)), np.squeeze(mu2 + 2*np.sqrt(var2)), alpha=0.4)
axes[1].plot(XS, mu2)
axes[1].scatter(X, Y2)

plt.show()

