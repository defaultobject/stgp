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

kern = Matern32(lengthscales=[1.0])
kern_batch = Matern32(lengthscales=[1.0])

m = lego.models.GP(
    X, Y, kernel=kern, inference='Markov'
)

m_batch = lego.models.GP(
    X, Y, kernel=kern_batch, inference='Batch'
)

restore = True
if restore:
    m.load_from_checkpoint(str(checkpoint_folder / 'sde_gp'))
    m_batch.load_from_checkpoint(str(checkpoint_folder / 'sde_gp_batch'))
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

    m.checkpoint(str(checkpoint_folder / 'sde_gp'))
    m_batch.checkpoint(str(checkpoint_folder / 'sde_gp_batch'))

    print(np.array(learning_curve_batch) - np.array(learning_curve))

    plt.plot(learning_curve, label='sde')
    plt.plot(learning_curve_batch, label='gp')
    plt.legend()
    plt.show()

print(kern.lengthscales, kern_batch.lengthscales)

XS = np.linspace(-1, 2, 500)[:, None]
#XS = X
mu, var = m.predict_f(XS)

mu_batch, var_batch = m_batch.predict_f(XS)

plt.fill_between(
    np.squeeze(XS), np.squeeze(mu_batch) - np.squeeze(2*np.sqrt(var_batch)), np.squeeze(mu_batch) + np.squeeze(2*np.sqrt(var_batch)), alpha=0.4
)
plt.plot(XS, mu_batch)

plt.fill_between(
    np.squeeze(XS), np.squeeze(mu) - np.squeeze(2*np.sqrt(var)), np.squeeze(mu) + np.squeeze(2*np.sqrt(var)), alpha=0.4
)
plt.plot(XS, mu)
plt.scatter(X, Y)
plt.show()
