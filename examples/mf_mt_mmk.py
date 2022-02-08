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

checkpoint_id = __file__

def train_adam(m_arr, epochs, plot=True):
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m_arr, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )
    if plot:
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

def generate_data():
    np.random.seed(0)

    N = 100
    x = np.linspace(0, 1, N)
    X = x[:, None]

    # Genenate low fidelity data
    lf_1_f = -np.sin(x*20)
    lf_1 =  lf_1_f + 1e-1*np.random.randn(N)

    lf_2_f = 2+3*np.sin(x*20)
    lf_2 =  lf_2_f + 1e-1*np.random.randn(N)

    # Generate high fidelity data
    hf_1 = lf_1_f*-1 + 1e-3*np.random.randn(N)
    hf_2 = np.log(1+np.abs(np.min(lf_2_f)) + lf_2_f) + 1e-3*np.random.randn(N)

    hf_2[20:60] = np.NaN

    if False:
        plt.plot(x, lf_1, label='lf_1')
        plt.plot(x, lf_2, label='lf_2')
        plt.plot(x, hf_1, label='hf_1')
        plt.plot(x, hf_2, label='hf_2')
        plt.show()
        exit()
    data =  {
        'X': X,
        'XS': X,
        'lf': {
            '1': lf_1[:, None],
            '2': lf_2[:, None],
        }, 
        'hf': {
            '1': hf_1[:, None],
            '2': hf_2[:, None],
        }
    }
    data['lf']['Y'] = np.hstack([
       data['lf']['1'], data['lf']['2']
    ])

    data['hf']['Y'] = np.hstack([
       data['hf']['1'], data['hf']['2']
    ])

    return data

data = generate_data()

def mt_hf_only(data, model=None, restore=False):
    P = 2
    Q = 2


    X = data['X']
    Y = np.hstack([
        data['hf']['1'], 
        data['hf']['2'], 
    ])

    # Construct independt prior
    latents = lego.transforms.Independent([
        lego.models.GP(X=X, kernel=RBF(input_dim=1, lengthscales=[0.1]), latent=True)
        for q in range(Q)
    ], prior=True)

    # Make LMC prior
    if model == 'lmc':
        print('lmc')
        name = 'lmc_hf_only'
        prior = lego.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)

    elif model=='ind-gp':
        print('independent GP')
        name = 'ind_gp_only'
        prior = latents
    else:
        raise RuntimeError(f'Model not found {model}')

    lik = [lego.likelihood.Gaussian(0.1) for p in range(P)]

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
        train_adam(m, 1000, True)
        m.checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))

    mu, var = m.predict_f(data['XS'])

    return {'mu': mu, 'var': var}

def mf_mt(data, model=None, restore=False):
    P = 2
    Q = 2


    X = data['X']
    Y_lf = data['lf']['Y']
    Y_hf = data['hf']['Y']

    # Construct LF models
    m_lf_list = [
        lego.models.GP(
            X=X, Y = Y_lf[:, i][:, None], kernel=RBF(input_dim=1, lengthscales=[0.1])
        )
        for i in range(Q)
    ]

    # pretrain
    if False:
        for m_lf in m_lf_list:
            train_adam(m_lf, 100, True)

    # construct MT HF model
    latents = lego.transforms.Independent([
        lego.models.GP(
            X=X, kernel=DeepRBF(parent=m_lf_list[q], lengthscale=[0.1])
        )
        for q in range(Q)
    ])

    # Make LMC prior
    if model == 'lmc':
        print('lmc')
        name = 'lmc_mf'
        prior = lego.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)

    elif model=='ind-gp':
        print('independent GP')
        name = 'ind_gp_mf'
        prior = latents
    else:
        raise RuntimeError(f'Model not found {model}')

    lik = [lego.likelihood.Gaussian(0.1) for p in range(P)]

    # Construct GP Model
    m = lego.models.GP(
        X=X, 
        Y = Y_hf,  
        prior = prior,
        inference='Batch', 
        likelihood=lik
    )

    m_arr = [m] + m_lf_list


    if restore:
        m.load_from_checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))
    else:
        train_adam(m_arr, 1000, True)
        m.checkpoint(str(checkpoint_folder / f'{checkpoint_id}_{name}'))

    mu, var = m.predict_f(data['XS'])

    return {'mu': mu, 'var': var}




results = {
    'lmc_hf_only': mt_hf_only(data, model='lmc', restore=False),
    'ind_hf_only': mt_hf_only(data,  model='ind-gp', restore=False),
    'ind_mf': mf_mt(data,  model='ind-gp', restore=False),
    'lmc_mf': mf_mt(data,  model='lmc', restore=False),
}

XS = data['XS']
X = data['X']
Y = data['hf']['Y']
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
