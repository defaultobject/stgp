""" Variational Gaussian Process Regression """
import sys
sys.path.append('../')

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", True)

import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data
from stgp.transforms import Independent
from stgp import settings
from tqdm import trange

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

# Construct Model
data = Data(X, Y)

m = GP(
    data = data,
    prior = Independent([
        GP(
            sparsity = stgp.sparsity.NoSparsity(Z = data.X), 
            kernel = ScaleKernel(RBF(input_dim=1, lengthscales=[0.1]), 0.2),
            prior = True
        )
    ]),
    likelihood = ProductLikelihood([Gaussian(0.1)]),
    inference='Variational'
)

# Train
if True:
    ng_trainer = NatGradTrainer(m)
    #ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()

    ng_trainer.train(1.0, 1)

pred_mu, pred_var = m.predict_f(XS)
pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)
breakpoint()

plt.fill_between(
    np.squeeze(XS), 
    np.squeeze(pred_mu - 1.96*np.sqrt(pred_var)), 
    np.squeeze(pred_mu + 1.96*np.sqrt(pred_var)), 
    facecolor=colors.LINE_COL, 
    alpha=0.3
)
plt.plot(XS, pred_mu, color=colors.LINE_COL, label='GP Fit')
plt.scatter(X, Y, color='black', label='Training Data')
plt.legend()
plt.show()

