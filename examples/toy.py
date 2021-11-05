import jax
import objax
import numpy as np
import pandas as pd
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)

import gpax
from gpax.trainers import SimpleTrainer, ScipyTrainer
from gpax.trainers.callbacks import progress_bar_callback
from gpax.kernels.deep_kernels import DeepRBF
from gpax.kernels import RBF
from gpax.models import GP

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(linewidth=200)

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

if False:
    P = 1
    Y_all = Y_all[:, 1][:, None]
    Y = Y[:, 1][:, None]

XS = np.linspace(0, 1, 1000)[:, None]

if False:
    plt.plot(X, Y[:, 0])
    plt.plot(X, Y[:, 1])
    plt.show()

    breakpoint()

print(f'X: {X.shape}, Y: {Y.shape}')

# Set up model

models = ['LMC', 'LMC_Unit_Tri', 'LMC_Corr', 'LMC_Corr_var', 'Independent']

res = {}

for model in models:

    P = Y.shape[1]
    Q = P
    latents = [
        GP(X=X, kernel=RBF(input_dim=1, lengthscales=[0.1]), latent=True)
        for q in range(Q)
    ]
    if model == 'LMC':
        prior = gpax.transforms.multi_output.LMC(latents, output_dim = P)
    elif model == 'LMC_Corr':
        prior = gpax.transforms.multi_output.LMC_Corr(latents, output_dim = P)
    elif model == 'LMC_Corr_var':
        prior = gpax.transforms.multi_output.LMC_Corr_var(latents, output_dim = P)
    elif model == 'LMC_Unit_Tri':
        prior = gpax.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)
    elif model == 'Independent':
        prior = gpax.transforms.basic.Independent(latents)

    m = GP(X=X, Y = Y, prior=prior, inference='Batch')

    # Train model

    if False:
        epochs = 1000

        callback = progress_bar_callback(epochs)

        learning_curve, training_time = ScipyTrainer().train(
            m, 
            'BFGS',
            0.01,
            epochs,
            callback = callback
        )

        if False:
            plt.plot(learning_curve)
            plt.show()

        m.checkpoint(f'toy_{model}')
    else:
        m.load_from_checkpoint(f'toy_{model}')
        pass

    print(f'lik: {m.likelihood[0].variance}')
    pred_mu, pred_var = m.predict(XS)
    W = prior.W
    print(f'lik: {m.likelihood[0].variance}')


    res[model] = {}
    res[model]['pred_mu'] = pred_mu
    res[model]['pred_var'] = pred_var
    res[model]['W'] = W

if True:
    num_models = len(models)
    fig, axes =plt.subplots(num_models, 2, squeeze=False)

    for m, model in enumerate(models):

        pred_mu = res[model]['pred_mu'] 
        pred_var = res[model]['pred_var'] 

        pred_mu = np.reshape(pred_mu, [P, XS.shape[0]])
        pred_var = np.reshape(pred_var, [P, XS.shape[0]])

        colors = cm.rainbow(np.linspace(0, 1, P))

        axes[m][0].set_title(model)

        for p in range(P):
            axes[m][0].fill_between(
                np.squeeze(XS), 
                pred_mu[p]-2*np.sqrt(pred_var[p]), 
                pred_mu[p]+2*np.sqrt(pred_var[p]), 
                alpha=0.1, 
                facecolor = colors[p]
            )
            axes[m][0].plot(XS, pred_mu[p], c=colors[p])
            axes[m][0].scatter(X, Y_all[:, p], c=colors[p])
            axes[m][0].scatter(X, Y[:, p], c=colors[p], edgecolors='black')

            axes[m][0].axvline(X[missing_region[0]])
            axes[m][0].axvline(X[missing_region[1]])

        mixing_matrix = res[model]['W']
        W = mixing_matrix @ mixing_matrix.T

        axes[m][1].imshow(W)

    plt.show()
