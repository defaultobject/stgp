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

checkpoint_id = 'hetreoskedastic'

# Load Data and construct datasets
data_path = Path('data') / 'motorcycle_data.txt'

f = open(data_path, "r")
tmp = f.read().split('\n')
data = []
for i in tmp:
    tmp2 = i.split('\t')
    data.append(tmp2)
data = np.array(data,dtype = float)

X = data[:,0][:, None]
Y = data[:,1][:, None]

Y = Y-np.mean(Y)
Y = Y/np.std(Y)

XS = np.linspace(np.min(X[:, 0])-20, np.max(X[:, 0])+20, 500)[:, None]

# Construct Models

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


def gp(X, Y, XS, train_fn, name, restore=False):
    epochs = 100

    m = lego.models.GP(
        X=X, 
        Y = Y,  
        kernel = lego.kernels.RBF(lengthscales=[5.0]),
        inference='Batch', 
        likelihood=lego.likelihood.Gaussian(1.0)
    )

    if restore:
        m.load_from_checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))
    else:
        train_fn(m, epochs)
        m.checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))

    mu, var = m.predict(XS, diagonal=True)

    return {'mu': mu, 'var': var}


def hetro_gp(X, Y, XS, train_fn, name, model_type, restore=False):
    epochs = 1000

    # Construct uncertain input GP
    #    Subsample data to avoid overfitting
    subsample = 10
    latent_noise_gp = lego.models.GP(
        X=X[::subsample, :], 
        Y = Y[::subsample, :], 
        latent_y=True, 
        inference='Batch',
        kernel = RBF(lengthscales=[1.0], variance=1.0)
    )

    # Construct Deep Kernel
    if model_type == 'prior':
        kern = RBF(lengthscales=[1.0], variance=0.1)*DeepRBF(DeepHetreo(latent_noise_gp))
    elif model_type == 'lik':
        # This is v. sensitive to how latent_noise_gp noise GP is init.
        kern = RBF(lengthscales=[1.0], variance=0.1) + DeepHetreo(latent_noise_gp)

    m = lego.models.GP(
        X=X, 
        Y = Y,  
        kernel = kern,
        inference='Batch', 
        likelihood=lego.likelihood.Gaussian(0.01)
    )

    train_fn(m, epochs)

    if restore:
        m.load_from_checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))
    else:

        m.checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))


    mu, var = m.predict(XS, diagonal=True)

    return {'mu': mu, 'var': var}



results = {
    'gp_bfgs': gp(X, Y, XS, train_bfgs, 'gp_bfgs', restore=False),
    'hetreo_gp_bfgs': hetro_gp(X, Y, XS, train_bfgs, 'hetreo_gp_bfgs', model_type='prior', restore=False),
    'hetreo_lik_gp_bfgs': hetro_gp(X, Y, XS, train_bfgs, 'hetreo_lik_gp_bfgs', model_type='lik', restore=False),
}

num_models = len(results.keys())

colors = cm.rainbow(np.linspace(0, 1, num_models))

fig, axes = plt.subplots(num_models, squeeze=False)

for i, model_name in enumerate(list(results.keys())):
    NS = XS.shape[0]
    mu_i = results[model_name]['mu'].reshape([NS])
    var_i = results[model_name]['var'].reshape([NS])
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
    axes[i][0].scatter(X, Y, c='black')

    axes[i][0].set_title(model_name)

plt.show()
