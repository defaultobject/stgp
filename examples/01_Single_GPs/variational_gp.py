""" Batch Gaussian Process Regression """
import sys
sys.path.append('../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF 
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data
from stgp.transforms import Independent
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
            kernel = RBF(input_dim=1, lengthscales=[0.1]),
            prior = True
        )
    ]),
    likelihood = ProductLikelihood([Gaussian()]),
    inference='Variational',
    ell_samples=100,
    prediction_samples=1000
)

# Train
if True:
    max_iters = 100

    ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()

    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    ng_trainer.train(1.0, 1)
    for i in trange(max_iters):
        trainer.train(1.0, 1)
        ng_trainer.train(0.1, 1)


pred_mu, pred_var = m.predict_y(XS)

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

