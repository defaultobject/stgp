import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp

import objax
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.kernels.deep_kernels import DeepRBF, DeepHetreo
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import RBF

checkpoint_folder = Path('checkpoints')
checkpoint_folder.mkdir(exist_ok=True)

checkpoint_id = 'multi_task'

def toy_data():
    np.random.seed(0)

    def nonstationary_function(x, seed=0):
        # See https://www.ics.uci.edu/~pazzani/Publications/ssdb99.pdf
        #   or https://archive.ics.uci.edu/ml/datasets/Pseudo+Periodic+Synthetic+Time+Series
        f = np.zeros(x.shape[0])
        np.random.seed(seed)
        for i in range(3, 7):
            rand_i = np.random.uniform(low=0.0, high=np.power(2, i), size=(1,))
            _f = (1/np.power(2, i)) * np.sin( 2*np.pi*( np.power(2,2+i)+rand_i) * x * 3.0)
            f += _f
        return f

    x = np.linspace(0, 1, 100)
    X = x[:, None]

    u1 = nonstationary_function(10*x, seed=0)[:, None]
    u2 = nonstationary_function(x, seed=2)[:, None]

    W = np.array([[2, 0.6], [0.6, 1.0]])

    Y_all = (W @ np.hstack([u1, u2]).T).T

    missing_region = [40, 60]

    Y = Y_all.copy()
    Y[missing_region[0]:missing_region[1], 0] = np.NaN

    XS = np.linspace(0, 1, 1000)[:, None]

    return X, Y, Y_all, XS

def train_adam(m, epochs):
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )
    if True:
        plt.plot(learning_curve)
        plt.show()

def train_bfgs(m, epochs):
    learning_curve, training_time = ScipyTrainer().train(
        m, 
        'BFGS',
        0.01,
        epochs,
        callback = None
    )


def lmc(X, Y, train_fn, name, restore=False):
    epochs = 500
    P = Y.shape[1]
    Q = P

    # Construct independt prior
    latents = [
        lego.models.GP(X=X, kernel=RBF(input_dim=1, lengthscales=[0.1]), latent=True)
        for q in range(Q)
    ]

    # Make LMC prior
    prior = lego.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)

    # Likelihood for each output
    lik = [lego.likelihood.Gaussian(1.0) for p in range(P)]

    # Construct GP Model
    m = lego.models.GP(
        X=X, 
        Y = Y,  
        prior = prior,
        inference='Batch', 
        likelihood=lik
    )

    if restore:
        m.load_from_checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))
    else:
        train_fn(m, epochs)
        m.checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))

    mu, var = m.predict(XS, diagonal=True)

    return {'mu': mu, 'var': var}

X, Y, Y_all, XS = toy_data()


results = {
    'lmc_bfgs': lmc(X, Y, train_adam, 'lmc_bfgs', restore=True)
}

num_models = len(results.keys())
P = Y.shape[1]

colors = cm.rainbow(np.linspace(0, 1, num_models))

fig, axes = plt.subplots(num_models, squeeze=False)

for i, model_name in enumerate(list(results.keys())):
    NS = XS.shape[0]

    for p in range(P):
        mu_i = results[model_name]['mu'][p].reshape([NS])
        var_i = results[model_name]['var'][p].reshape([NS])
        XS_i = np.squeeze(XS)

        axes[i][0].fill_between(
            XS_i, 
            mu_i-1.96*np.sqrt(var_i), 
            mu_i+1.96*np.sqrt(var_i), 
            facecolor=colors[i],
            alpha=0.4
        )

        axes[i][0].plot(
            XS_i, 
            mu_i, 
            c=colors[i]
        )
        axes[i][0].scatter(X, Y[:, p], c='black')

    axes[i][0].set_title(model_name)

plt.show()
