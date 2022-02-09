import jax
from jax import make_jaxpr
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import RBF, ScaleKernel, BiasKernel

import objax
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
name = __file__

# generate data
P = 1
N = 50


xs = np.linspace(-0.5, 1.5, 1000)
x = np.linspace(0, 1, N)
y = np.sin(x*10)+0.1*np.random.randn(N)

XS = xs[:, None]
X = x[:, None]
Y = y[:, None]


m1 = lego.models.GP(X, Y, kernel=ScaleKernel(RBF(lengthscales=[0.1])))


model_list = [m1]

epochs = 500

restore = False

if True:
    if restore:
        m2.load_from_checkpoint(str(checkpoint_folder / name))
    else:
        callback = progress_bar_callback(epochs)
        learning_curve, training_time = SimpleTrainer().train(
            model_list, 
            objax.optimizer.Adam,
            0.01,
            epochs,
            callback = callback
        )
        m1.checkpoint(str(checkpoint_folder / name))

        plt.plot(learning_curve)
        plt.show()

mu1, var1 = m1.predict_y(XS)

fig, axes = plt.subplots(1, 1)

axes.fill_between(np.squeeze(XS), np.squeeze(mu1 - 2*np.sqrt(var1)), np.squeeze(mu1 + 2*np.sqrt(var1)), alpha=0.4)
axes.plot(XS, mu1)
axes.scatter(X, Y1)

plt.show()

