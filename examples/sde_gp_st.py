import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import RBF, ScaleKernel, BiasKernel, Matern32, SpatioTemporalSeperableKernel
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

from sklearn.datasets import make_blobs

checkpoint_folder = Path('checkpoints')
checkpoint_folder.mkdir(exist_ok=True)

def create_grid(x1, x2, y1, y2, n1=10, n2=10):
    y = np.linspace(y1, y2, n2)
    x = np.linspace(x1, x2, n1)

    grid = []
    for i in x:
        for j in y:
            grid.append([i, j])

    return np.array(grid)


Nt_train = 100
Ns = 8

X = create_grid(-1, 1, -1, 1, Nt_train, Ns)
N = X.shape[0]

np.random.seed(0)
y = np.sin(10*X[:, 0]) + np.sin(10*X[:, 1]) + 0.01*np.random.randn(N)
Y = y[:, None]

Y[10:40, :] = np.NaN

nan_idx = np.isnan(Y[:, 0])

# generate data

kern = SpatioTemporalSeperableKernel(
    Matern32(lengthscales=[0.1], active_dims=[0], input_dim=1),
    Matern32(lengthscales=[0.1], active_dims=[1], input_dim=1)
)

kern_batch = Matern32(lengthscales=[0.1, 0.1], input_dim=2)

m = lego.models.GP(
    X, Y, kernel=kern, inference='Markov'
)

m_batch = lego.models.GP(
    X[~nan_idx], Y[~nan_idx], kernel=kern_batch, inference='Batch'
)

print('sde: ', m.get_objective())
print('batch: ', m_batch.get_objective())

#plt.figure(figsize=(20, 10))
#plt.scatter(X[:, 0], X[:, 1], c=Y)
#plt.show()
#exit()

restore = True
if restore:
    m.load_from_checkpoint(str(checkpoint_folder / 'sde_gp_st'))
    m_batch.load_from_checkpoint(str(checkpoint_folder / 'sde_gp_batch_st'))
else:
    epochs = 100
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )

    callback = progress_bar_callback(epochs)

    learning_curve_batch, training_time = SimpleTrainer().train(
        m_batch, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )

    m.checkpoint(str(checkpoint_folder / 'sde_gp_st'))
    m_batch.checkpoint(str(checkpoint_folder / 'sde_gp_batch_st'))

    print(np.array(learning_curve_batch) - np.array(learning_curve))

    plt.plot(learning_curve, label='sde')
    plt.plot(learning_curve_batch, label='gp')
    plt.legend()
    plt.show()

print('sde: ', m.get_objective())
print('batch: ', m_batch.get_objective())


XS = create_grid(-1, 1, -1, 1, 100, 100)
mu, var = m.predict_f(XS)
mu_batch, var_batch = m_batch.predict_f(XS)

mu, var = np.squeeze(mu), np.squeeze(var)
mu_batch, var_batch = np.squeeze(mu_batch), np.squeeze(var_batch)

print(mu_batch)
print(mu)

print( np.sum((mu_batch-mu)**2), ' ', np.sum((var_batch-var)**2))

fig, axes = plt.subplots(2, 1)
axes[0].scatter(XS[:, 0], XS[:, 1], c=mu)
axes[1].scatter(XS[:, 0], XS[:, 1], c=mu_batch)

plt.show()
