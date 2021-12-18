import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import RBF, ScaleKernel, BiasKernel, Matern32
from legogp.computation.filtering import sequential_kalman_filter, sequential_rts_smoother, filter_and_smooth
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
N = 100
x = np.linspace(0, 1, N)
y = -np.sin(x*8)+0.01*np.random.randn(N)

X = x[:, None]
Y = y[:, None]

kern = Matern32(lengthscales=[0.1])

m = lego.models.GP(
    X, Y, kernel=kern, inference='Markov'
)

epochs = 1000
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

mu, var = m.predict_f(X)

plt.fill_between(
    np.squeeze(X), np.squeeze(mu) - np.squeeze(2*np.sqrt(var)), np.squeeze(mu) + np.squeeze(2*np.sqrt(var)), alpha=0.4
)
plt.plot(X, mu)
plt.scatter(X, Y)
plt.show()
