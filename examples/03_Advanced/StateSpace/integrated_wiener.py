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
train = False

np.random.seed(0)

# Construct Data
_, X, Y = single_output_timeseries(200, 100, seed=0)
X = np.linspace(0, 2, 200)[:, None]
Y = np.zeros_like(X)

#XS = np.linspace(0, np.max(X)*0.5, 1000)[:, None] # only defined for t >= 0
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
    lik = ReshapedGaussian(Gaussian(variance=0.00001), num_blocks=data.Nt, block_size=1)
    latent_gp = GP(
        sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
        kernel =  WienerVelocity(q=q, variance=var),
        #kernel =  Matern32(lengthscales=[1.0]),
        prior = True
    )
    #prior = IdentityPDE(LTI_SDE(Independent([latent_gp])), m_init = [[10.0], [10.0]])
    prior = SimpleODE(LTI_SDE(Independent([latent_gp])), m_init=[2.0, 0.0])
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

m.prior.pred_mode = False

if False:
    pred_mu, pred_var = m.predict_f(XS, filter_only=True)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)
    pred_mu = pred_mu[:, 0]
    pred_var = pred_var[:, 0]
else:
    pred_mu, pred_var = m.predict_f(XS)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)

print(np.sum(pred_mu), np.sum(pred_var))

plt.figure()
plt.fill_between(np.squeeze(XS), pred_mu - 1.96*np.sqrt(pred_var), pred_mu + 1.96*np.sqrt(pred_var), alpha=0.4)
plt.plot(XS, pred_mu)
plt.scatter(X, Y)
plt.show()
