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
from stgp.trainers import ScipyTrainer, GradDescentTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels.matern import ScaledMatern32

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

# Construct Model
# Will use default RBF kernel and Gaussian Likelihood
m = GP(X, Y, kernel=[ScaledMatern32(input_dim=1, lengthscales=[0.1], variance=0.2)])

pred_mu, pred_var = m.predict_y(XS)

# Train
print(m.get_objective())
if False:
    max_iters = 100
    trainer = ScipyTrainer(m, 'CG')
    trainer.train(None, max_iters, callback=progress_bar_callback(max_iters))

print('NLPD: ', m.nlpd(X, Y))

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

