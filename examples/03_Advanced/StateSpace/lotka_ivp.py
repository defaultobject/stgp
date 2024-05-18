""" Integrated Wiener Process  State Space Representation for the Lotka-Volterra Model"""

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors

import stgp
from stgp.models import GP
from stgp.kernels import IntegratedWiener, Wiener, WienerVelocity, Matern32
from stgp.likelihood import Gaussian, ReshapedGaussian, DiagonalGaussian
from stgp.trainers.standard import LBFGS
from stgp.trainers.callbacks import progress_bar_callback
from stgp.data import TemporalData, MultiOutputTemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms.pdes import  LotkaVolterra

import matplotlib.pyplot as plt
q = 1
var = 1.0
train = False

#X = np.linspace(0, 40, 2000)[:, None]
X = np.arange(0, 40, 0.01)[:, None]
Y = np.zeros_like(X)
Y = np.hstack([Y, Y, Y, Y])*np.NaN
XS = np.linspace(0, np.max(X), 2000)[:, None]

print(X.shape, Y.shape, XS.shape)
data = MultiOutputTemporalData(X, Y)
lik = ReshapedGaussian(DiagonalGaussian(variance=[1e-3, 1e-3, 1e-3, 1e-3]), num_blocks=data.Nt, block_size=4)
#lik = ReshapedGaussian(Gaussian(variance=[1e-3, 1e-3]), num_blocks=data.Nt, block_size=2)

latent_gp = Independent([
    GP(
        sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
        kernel =  IntegratedWiener(q=q, variance=var),
        prior = True
    )
    for i in range(2)
])

# must be the same shape and ordering as Y
#boundary_conditions = np.array([[ 20.0, 0.0,  20.0,   0.0 ]])
boundary_conditions = np.array([[ 20.0,  10.0, 20.0, 10.0]])
boundary_conditions = np.vstack([boundary_conditions, np.ones([data.Nt-1, boundary_conditions.shape[1]])*np.NaN])[..., None]

if False:
    prior = LotkaVolterra(LTI_SDE(latent_gp), 0.5, 0.05, 0.05, 0.5, [20.0, 10.0, 20.0, 10.0], boundary_conditions=boundary_conditions*np.NaN, full_state=True, boundary_by_init=True)
else:
    prior = LotkaVolterra(LTI_SDE(latent_gp), 0.5, 0.05, 0.05, 0.5, boundary_conditions=boundary_conditions, train_m_init = False, full_state = True, boundary_by_init=False)

m = GP(
    data = data,
    prior = prior,
    likelihood = lik,
    inference='Sequential',
    full_state = True
)

pred_mu, pred_var = m.predict_f(XS, filter_only=False, force_full_state=True)

print(m.get_objective())

if train:
    lc, _ = LBFGS(m).train(None, 100, callback=progress_bar_callback(100))
    plt.plot(lc)
    plt.show()

#pred_mu, pred_var = m.predict_f(XS, filter_only=True)
pred_mu, pred_var = m.predict_f(XS, filter_only=False, force_full_state=True)
pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)

y1_grad = np.gradient(pred_mu[:, 0], np.squeeze(XS))
y2_grad = np.gradient(pred_mu[:, 2], np.squeeze(XS))

print(pred_mu.shape)

if True:
    Q = 4
    fig, ax = plt.subplots(Q, 1)
    for i in range(Q):
        ax[i].fill_between(np.squeeze(XS), pred_mu[:, i] - 1.96*np.sqrt(pred_var[:, i]), pred_mu[:, i] + 1.96*np.sqrt(pred_var[:, i]), alpha=0.4)
        ax[i].plot(XS, pred_mu[:, i], label='learnt')

    ax[1].plot(XS, y1_grad, linestyle='dashed')
    ax[3].plot(XS, y2_grad, linestyle='dashed')
    plt.show()


import stgp
from stgp.computation.solvers.euler import euler
from stgp.transforms.latent_force import LotkaVolterra
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

np.random.seed(0)

alpha = 0.5
beta = 0.05
delta = 0.05
gamma = 0.5
init_state = [20.0, 20.0]

N = 500000
dt = 0.0001

x = np.cumsum(np.ones(N) * dt)

m_lfm = LotkaVolterra(None, alpha, beta, delta, gamma, init_state = init_state)
res, init_x = euler(m_lfm, np.array(init_state), N, dt)

plt.figure()
for i in [0, 2]:
    plt.fill_between(np.squeeze(XS), pred_mu[:, i] - 1.96*np.sqrt(pred_var[:, i]), pred_mu[:, i] + 1.96*np.sqrt(pred_var[:, i]), alpha=0.4)
    plt.plot(XS, pred_mu[:, i], label='learnt')

plt.plot(x, init_x[:, 0], linestyle='dashed')
plt.plot(x, init_x[:, 1], linestyle='dashed')

plt.show()




