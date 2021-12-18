import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import RBF, ScaleKernel, BiasKernel

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

from legogp.computation.gaussian import log_gaussian

checkpoint_folder = Path('checkpoints')
checkpoint_folder.mkdir(exist_ok=True)

# generate data
P = 10

N = 50


XS = np.linspace(-0.5, 1.5, 1000)[:, None]
x = np.linspace(0, 1, N)
y1 = np.sin(x*10)+0.0001*np.random.randn(N)+2.0

X = x[:, None]
Y1 = y1[:, None]

K_rbf = ScaleKernel(RBF())
K_bias = ScaleKernel(BiasKernel())
K = K_rbf + K_bias

m2 = lego.models.GP(X, Y1, kernel=K)

model_list = [m2]


epochs = 1000

restore = True

if restore:
    m2.load_from_checkpoint(str(checkpoint_folder / 'mf'))
else:
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        model_list, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )
    m2.checkpoint(str(checkpoint_folder / 'mf'))

    plt.plot(learning_curve)
    plt.show()

mu2, var2 = m2.predict_f(XS)

def plot_param_landscape():
    Y = Y1
    N = Y.shape[0]
    m = m2

    #K_xx = np.array(m.prior.latents[0].kernel.K(X, X))
    K_xx = K_rbf.K(X, X)
    Lambda = m.likelihood[0].variance * np.eye(N)

    def nll(bias):
        return -float(log_gaussian(Y, np.zeros_like(Y), K_xx + Lambda + bias*np.ones([N, N])))

    x = np.linspace(4, 5, 100000)
    nll_arr = [nll(x) for x in x] 

    print('Min at: ', x[np.argmin(nll_arr)])

    return x[np.argmin(nll_arr)]

    #plt.plot(x, nll_arr)
    #plt.show()


def _optimal_bias():
    Y = Y1
    N = Y1.shape[0]

    sigma = K_rbf.K(X, X) + m2.likelihood[0].variance*np.eye(N) + 1e-5 * np.eye(N)

    return np.sum(Y @ Y.T - sigma)/(N**2)

def optimal_bias():
    Y = Y1
    N = Y1.shape[0]

    sigma = K_rbf.K(X, X) + m2.likelihood[0].variance*np.eye(N)
    sigma_inv = np.linalg.inv(sigma)
    ones = np.ones([N, N])
    YY = Y @ Y.T

    a = np.sum(sigma_inv @ ones @ sigma_inv @ YY @ sigma_inv @ ones @ sigma_inv)
    b = np.sum(sigma_inv @ ones @ sigma_inv - sigma_inv @ ones @ sigma_inv @ YY @ sigma_inv  - sigma_inv @ YY @ sigma_inv @ ones @ sigma_inv )
    c = np.sum(sigma_inv @ YY @ sigma_inv  - sigma_inv)


    roots = np.roots([a, b, c])

    def find_a(r):
        e = np.ones([N, 1])
        return r / (1 - r * e.T @ sigma_inv @ e)

    bias = [find_a(r) for r in roots]
    print(roots)
    print(bias)

    breakpoint()

    return bias[1]

found_optimal_bias = plot_param_landscape()
bias = float(K_bias.variance)
pred = float(mu2[0])
avg = np.mean(Y1)
optimal_bias = optimal_bias()


print(f'bias: {bias}, pred: {pred}, optimal: {optimal_bias}, found optimal: {found_optimal_bias}, avg: {avg}')


breakpoint()

fig, axes = plt.subplots(1, 1)

axes.fill_between(np.squeeze(XS), np.squeeze(mu2 - 2*np.sqrt(var2)), np.squeeze(mu2 + 2*np.sqrt(var2)), alpha=0.4)
axes.plot(XS, mu2)
axes.scatter(X, Y1)

plt.show()

