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
from legogp.kernels import RBF, ScaleKernel

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

#XS = np.linspace(np.min(X[:, 0])-20, np.max(X[:, 0])+20, 500)[:, None]
XS = X

# Construct Models

def train_adam(m, epochs, ls=0.01):
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        ls,
        epochs,
        callback = callback
    )
    if True:
        print(learning_curve[0], learning_curve[-1])
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

    mu, var = m.predict_f(XS, diagonal=True)

    return {'mu': mu, 'var': var}

def vgp(X, Y, XS, train_fn, model, name, restore=False):
    epochs = 1000

    Q = 1
    P = 1
    f_latents = [
        lego.models.GP(X=X, kernel=ScaleKernel(RBF(input_dim=1, lengthscales=[5.0])), latent=True) for q in range(Q)
    ]

    W_latents = [
        [
            lego.models.GP(X=X, kernel=ScaleKernel(RBF(input_dim=1, lengthscales=[10.0])), latent=True) for q in range(Q)
        ] 
        for p in range(P)
    ]

    if model == 'gprn':
        prior = lego.transforms.multi_output.GPRN(W_latents, f_latents, output_dim = P)
    elif model == 'gprn-exp':
        prior = lego.transforms.multi_output.GPRN_Exp(W_latents, f_latents, output_dim = P)

    q = lego.approximate_posteriors.FullGaussianApproximatePosterior(dim=2 * X.shape[0])

    m = lego.models.GP(
        X=X, 
        Y = Y,  
        prior=prior,
        inference='Variational', 
        approximate_posterior=q,
        likelihood=[lego.likelihood.Gaussian(0.001)]
    )

    if restore:
        m.load_from_checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))
    else:
        train_fn(m, epochs, 0.01)
        m.checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))

    m.print()


    if True:
        mu_latent, var_latent = m.predict_latents(XS, diagonal=True)

        P = mu_latent.shape[0]

        colors = cm.rainbow(np.linspace(0, 1, P))
        for i in range(P):
            mu_i = np.squeeze(mu_latent[i])
            var_i = np.squeeze(var_latent[i])
            XS_i = np.squeeze(XS)

            plt.fill_between(
                XS_i, 
                mu_i-1.96*np.sqrt(var_i), 
                mu_i+1.96*np.sqrt(var_i), 
                facecolor=colors[i],
                alpha=0.4
            )
            plt.plot(XS_i, mu_i, label=i)

        plt.legend()
        plt.show()

    mu, var = m.predict_f(XS, diagonal=True)

    return {'mu': mu, 'var': var}



results = {
    #'gp_adam': gp(X, Y, XS, train_adam, 'gp_adam', restore=False),
    #'gprn_adam': vgp(X, Y, XS, train_adam, 'gprn', 'gprn_adam', restore=False),
    'gprn_exp_adam': vgp(X, Y, XS, train_adam, 'gprn-exp', 'gprn_exp_adam', restore=True),
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
