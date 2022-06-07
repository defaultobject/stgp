import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp
import objax

import numpy as np
import legogp as lego
from legogp.transforms import Independent
from legogp.transforms.latent_force import NonLinearLFM, LotkaVolterra, Linearized, PopulationLotkaVolterra, RM_Population
from legogp.transforms.sdes import LTI_SDE, EulerMaruyama
from legogp.kernels import Matern32, ScaledMatern32, ApproxSDEPeriodic
from legogp.models import GP
from legogp.sparsity import NoSparsity
from legogp.computation.solvers.euler import euler
from legogp.computation.filters import kalman_filter as kf
from legogp.computation.filters import rts_smoother as rts
from legogp.inference import StatisticallyLinearisedFilter
from legogp.data import Data, MultiOutputTemporalData
from legogp import settings
from legogp.core import Model
from legogp.likelihood import Gaussian, BlockDiagonalGaussian, ReshapedGaussian

from legogp.trainers import GradDescentTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

def get_data():
    data_path = Path('/Users/ohamelijnck/Documents/projects/turing-HY/code/datasets/lynx_hare/downloaded_data')
    df = pd.read_csv(data_path / 'lynx_hare.csv')
    return df

df = get_data()

if False:
    plt.plot(df['year'], df['lynx'], label='lynx')
    plt.plot(df['year'], df['hare'], label='hare')
    plt.legend()
    plt.show()

X = np.array(df['year'])[:, None]
X = (X - np.min(X)) 
Y = np.array(df[['hare', 'lynx']])

Y = np.log(Y)

X_vis = np.copy(X)
Y_vis = np.copy(Y)

Y[40:50, 1] = np.NaN

XS = np.linspace(np.min(X), np.max(X), 1000)[:, None]
YS = np.ones([XS.shape[0], 2])*np.NaN

X = np.vstack([X, XS])
Y = np.vstack([Y, YS])

Y = (Y - np.nanmean(Y, axis=0))/np.nanstd(Y, axis=0)


data = MultiOutputTemporalData(X=X, Y=Y)

latents = [
    GP(sparsity=NoSparsity(), kernel=ApproxSDEPeriodic(5, 10.0, 1.0, 15))
    for q in range(2)
]

likelihood = BlockDiagonalGaussian(2, 1, variance=1e-1*np.tile(np.eye(2), [1, 1, 1]))
likelihood = ReshapedGaussian(likelihood, data.Nt, 2)

base_gp = LTI_SDE(Independent(latents))

a = 3.2 
b = 0.6
c= 50.0
d= 0.56
k = 125.0
r = 1.6


alpha = -0.57
beta = -0.23
delta = -0.2
gamma = -0.17

#m_lfm = PopulationLotkaVolterra(base_gp, 3.2, 0.6, 50.0, 0.56, 125.0, 1.6, init_state = Y[0, :])
#m_lfm = LotkaVolterra(base_gp, alpha, beta, delta, gamma, init_state = Y[0, :])

if True:
    alpha = 5.0
    K = 0.8
    beta = 1.48
    b = 1e-5
    gamma = 12.41
    delta = 12.37

    m_lfm = RM_Population(base_gp, alpha, K, beta, b, gamma, delta, init_state = Y[0, :])

sde_m = EulerMaruyama(m_lfm)

likelihood.base.variance_param.fix()

m = GP(
    data = data,
    likelihood = likelihood,
    prior = sde_m,
    inference = 'Sequential'
)

if True:
    m.print()
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)
    #trainer = ScipyTrainer(m, 'L-BFGS-B')
    #trainer = ScipyTrainer(m, 'CG')

    epochs = 100
    callback = progress_bar_callback(epochs)
    learning_rates, _ = trainer.train(0.001, epochs, callback=callback)

    plt.plot(learning_rates)
    plt.show()
    m.print()

pred_mu, pred_var = m.predict_f(XS)

fig, axes = plt.subplots(2, 1)
for i in range(2):
    axes[i].fill_between(np.squeeze(XS), pred_mu[:, i] + np.sqrt(pred_var[:, 1]), pred_mu[:, i] - np.sqrt(pred_var[:, 1]), alpha=0.4)
    axes[i].plot(XS, pred_mu[:, i])
    axes[i].scatter(X_vis, Y_vis[:, i], c='black')
    axes[i].scatter(X, Y[:, i], c='grey')
plt.show()
