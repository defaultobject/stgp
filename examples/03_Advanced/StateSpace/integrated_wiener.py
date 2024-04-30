""" Integrated Wiener Process  State Space Representation"""

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors

import stgp
from stgp.models import GP
from stgp.kernels import IntegratedWiener, Wiener, WienerVelocity
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.trainers.standard import LBFGS
from stgp.trainers.callbacks import progress_bar_callback
from stgp.data import TemporalData


import matplotlib.pyplot as plt

batch = False
q = 1
var = 1.0

np.random.seed(0)

# Construct Data
_, X, Y = single_output_timeseries(50, 1000, seed=0)

XS = np.linspace(0, 1.5, 1000)[:, None] # only defined for t >= 0

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
    data = TemporalData(X, Y)
    lik = ReshapedGaussian(Gaussian(variance=0.01), num_blocks=data.Nt, block_size=1)
    m = GP(
        data = data,
        #kernel = IntegratedWiener(theta=1.0, q=1),
        #kernel = Wiener(q=1.0),
        kernel = WienerVelocity(q=q, variance=var),
        likelihood = lik,
        inference='Sequential'
    )

m.likelihood.fix()
m.print()
print(m.get_objective())

breakpoint()

if True:
    lc, _ = LBFGS(m).train(None, 100, callback=progress_bar_callback(100))
    plt.plot(lc)
    plt.show()
m.print()
print(m.get_objective())
breakpoint()

pred_mu, pred_var = m.predict_f(XS)
pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)

plt.figure()
plt.fill_between(np.squeeze(XS), pred_mu - 1.96*np.sqrt(pred_var), pred_mu + 1.96*np.sqrt(pred_var), alpha=0.4)
plt.plot(XS, pred_mu)
plt.scatter(X, Y)
plt.show()
