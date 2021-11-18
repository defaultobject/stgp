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


    N = 100
    x = np.linspace(0, 1, N)
    X = x[:, None]

    u1 = -np.sin(X*10)
    u2 = -np.sin(X*15)**2

    W = np.array([[2, 0.6], [0.6, 1.0]])

    Y_all = (W @ np.hstack([u1, u2]).T).T

    eps_1 = 0.1*np.random.rand(N)[:, None]
    eps_2 = 0.01*np.random.normal(scale=np.arange(0,N)/2)[:, None]
    eps = np.hstack([eps_1, eps_2])

    Y_all = Y_all + eps

    missing_region = [70, 85]

    Y = Y_all.copy()

    Y[missing_region[0]:missing_region[1], 0] = np.NaN

    if False:
        plt.plot(X, Y); 
        plt.show()
        exit()

    XS = np.linspace(0, 1, 1000)[:, None]

    return X, Y, Y_all, XS

def train_adam(m_arr, epochs):
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m_arr, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )
    if False:
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

def gp(X, Y, train_fn, name, model_type, restore=False):
    epochs = 500
    P = Y.shape[1]
    Q = P

    # Construct independt prior
    latents = [
        lego.models.GP(X=X, kernel=RBF(input_dim=1, lengthscales=[0.01]), latent=True)
        for p in range(P)
    ]

    # Make LMC prior
    prior = lego.transforms.Independent(latents=latents)


    # Likelihood for each output
    lik = [lego.likelihood.Gaussian(0.01) for p in range(P)]

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



def lmc(X, Y, train_fn, name, model_type, restore=False):
    epochs = 2000
    P = Y.shape[1]
    Q = P

    # Construct independt prior
    latents = [
        lego.models.GP(X=X, kernel=RBF(input_dim=1, lengthscales=[0.1]), latent=True)
        for q in range(Q)
    ]

    # Make LMC prior
    prior = lego.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)

    if model_type == 'LMC': 
        pass
    elif model_type == 'LMC_Hetreo':
        subsample = 15
        latent_noise_gp = [lego.models.GP(
                X=X[::subsample, :]+0.01, 
                Y = Y[::subsample, p][:, None]+p, 
                latent_y=True, 
                latent_x=False, 
                kernel=[RBF(lengthscales=[1.0]) for p in range(1)], 
                inference='Batch', 
                likelihood=[lego.likelihood.Gaussian(1.0) for p in range(1) ]
            )
            for p in range(P)
        ]

        prior_p = lego.transforms.Independent(
            latents=[lego.models.GP(X=X, kernel=[DeepHetreo(latent_noise_gp[p])], latent=True) for p in range(P)],
            prior=True
        )


        # Push through a deep kernel
        prior_p = lego.transforms.transform.DeepKernel_One2One(
            prior = prior_p,
            kernels = [DeepRBF(input_dim=1) for p in range(P)]
        )
        prior = lego.transforms.SumTransform(prior, prior_p)


    # Likelihood for each output
    lik = [lego.likelihood.Gaussian(0.1) for p in range(P)]

    # Construct GP Model
    m = lego.models.GP(
        X=X, 
        Y = Y,  
        prior = prior,
        inference='Batch', 
        likelihood=lik
    )

    if model_type == 'LMC_Hetreo':
        models = [m] + latent_noise_gp
    else:
        models = [m]

    if restore:
        m.load_from_checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))
    else:
        train_fn(models, epochs)
        m.checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))

    if model_type == 'LMC_Hetreo':
        for mod in latent_noise_gp:
            mu, var = mod.predict(XS, diagonal=True)

            NS = XS.shape[0]
            mu_i = mu.reshape([NS])
            var_i = var.reshape([NS])
            XS_i = np.squeeze(XS)
            plt.fill_between(XS_i, mu+np.sqrt(var), mu-np.sqrt(var), alpha=0.4)
            plt.plot(XS_i, mu)
            plt.scatter(mod.X.value, mod.Y.value)
            plt.show()

    mu, var = m.predict(XS, diagonal=True)

    return {'mu': mu, 'var': var}

X, Y, Y_all, XS = toy_data()


results = {
    #'gp_bfgs': gp(X, Y, train_bfgs, 'gp_bfgs', model_type = 'LMC', restore=False),
    #'lmc_bfgs': lmc(X, Y, train_bfgs, 'lmc_bfgs', model_type = 'LMC', restore=False),
    #'lmc_adam': lmc(X, Y, train_adam, 'lmc_adam', model_type = 'LMC', restore=False),
    #'lmc_hetro_bfgs': lmc(X, Y, train_bfgs, 'lmc_hetro_bfgs', model_type= 'LMC_Hetreo' , restore=False),
    'lmc_hetro_adam': lmc(X, Y, train_adam, 'lmc_hetro_adam', model_type= 'LMC_Hetreo' , restore=True)
}

num_models = len(results.keys())
P = Y.shape[1]


fig, axes = plt.subplots(num_models, squeeze=False)

colors = cm.rainbow(np.linspace(0, 1, P))

for i, model_name in enumerate(list(results.keys())):
    NS = XS.shape[0]

    for p in range(P):
        mu_i = results[model_name]['mu'][p].reshape([NS])
        var_i = results[model_name]['var'][p].reshape([NS])
        XS_i = np.squeeze(XS)

        #print(var_i)

        axes[i][0].fill_between(
            XS_i, 
            mu_i-1.96*np.sqrt(var_i), 
            mu_i+1.96*np.sqrt(var_i), 
            facecolor=colors[p],
            alpha=0.4
        )

        axes[i][0].plot(
            XS_i, 
            mu_i, 
            c=colors[p],
            label=f'O: {p}'
        )
        axes[i][0].scatter(X, Y[:, p], c='black')

    axes[i][0].set_title(model_name)

plt.show()
