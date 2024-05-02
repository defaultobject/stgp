""" Integrated Wiener Process  State Space Representation"""

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
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.trainers.standard import LBFGS
from stgp.trainers.callbacks import progress_bar_callback
from stgp.data import TemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms.pdes import IdentityPDE, Pendulum1D, SimpleODE


import matplotlib.pyplot as plt

batch = False
q = 1
var = 1.0
train = True
matern=False

np.random.seed(0)

# Construct Data
_, X, Y = single_output_timeseries(200, 100, seed=0)
#X = np.arange(0, 2, 0.01)[:, None]
X = np.linspace(0, 2, 10)[:, None]
Y = np.zeros_like(X)

XS = np.linspace(0, np.max(X), 1000)[:, None] # only defined for t >= 0
XS = X

if False:
    plt.scatter(X, Y)
    plt.show()

if batch:
    m = GP(
        X = X,
        Y = Y,
        #kernel = IntegratedWiener(theta=1.0, q=1),
        #kernel = Wiener(q=1.0),
        kernel = WienerVelocity(q=q, variance=var),
        likelihood = Gaussian(0.01)
    )

else:
    Y = Y*0.0
    data = TemporalData(X, Y)
    lik = ReshapedGaussian(Gaussian(variance=0.0), num_blocks=data.Nt, block_size=1)

    if matern:
        kern = Matern32(lengthscales=[1.0])
    else:
        kern = IntegratedWiener(q=q, variance=var)

    latent_gp = GP(
        sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
        kernel =  kern,
        prior = True
    )
    #prior = IdentityPDE(LTI_SDE(Independent([latent_gp])), m_init = [[10.0], [10.0]])
    prior = SimpleODE(LTI_SDE(Independent([latent_gp])), m_init=[1.0, 0.0])
    #prior = LTI_SDE(Independent([latent_gp]))

    m = GP(
        data = data,
        prior = prior,
        likelihood = lik,
        inference='Sequential'
    )

m.likelihood.fix()
m.print()
print(m.get_objective())

if train:
    lc, _ = LBFGS(m).train(None, 100, callback=progress_bar_callback(100))
    plt.plot(lc)
    plt.show()
m.print()

plot_grad = False
if True:
    pred_mu, pred_var = m.predict_f(XS, filter_only=True)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)
    pred_mu = pred_mu[:, 0]
    pred_var = pred_var[:, 0]
else:
    pred_mu, pred_var = m.predict_f(XS, force_full_state=True)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)

    grad = np.gradient(pred_mu[:, 0], np.squeeze(XS))

    if plot_grad:
        i = 1
    else: 
        i = 0
    pred_mu = pred_mu[:, i]
    pred_var = pred_var[:, i]



plt.figure()
plt.fill_between(np.squeeze(XS), pred_mu - 1.96*np.sqrt(pred_var), pred_mu + 1.96*np.sqrt(pred_var), alpha=0.4)
plt.plot(XS, pred_mu, label='learnt')
if plot_grad:
    plt.plot(np.linspace(0, 2, 100), np.linspace(0, 2, 100)*2, label='truth', linestyle='dashed')
    plt.plot(XS, grad, label='np grad', linestyle='dashed')
else:
    plt.plot(np.linspace(0, 2, 100), np.linspace(0, 2, 100)**2+2, label='truth', linestyle='dashed')
    plt.plot(XS, grad, label='np grad', linestyle='dashed')
plt.legend()
plt.plot()
plt.show()
