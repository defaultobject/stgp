""" Gaussian Process Regression Computed through a State Space Representation"""

import sys
sys.path.append('../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors

import stgp
from stgp.models import GP
from stgp.trainers import ScipyTrainer, GradDescentTrainer
from stgp.trainers.standard import ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32, ScaledMatern32, ScaledMatern52
from stgp.data import TemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent

import matplotlib.pyplot as plt

np.random.seed(0)

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

# Construct Model
data = TemporalData(X, Y)
lik = ReshapedGaussian(Gaussian(variance=1.0), num_blocks=data.Nt, block_size=1)
#kern = ScaledMatern32(input_dim=1, lengthscales=[1.0], variance=1.0)
kern = ScaledMatern52(input_dim=1, lengthscales=[1.0], variance=1.0)

latent_gp = GP(
    sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
    kernel = kern,
    prior = True
)
prior = LTI_SDE(Independent([latent_gp])) 

#m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential', filter_type='square_root_svm')
#m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential', filter_type='parallel')
m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential')

# Train
print(m.get_objective())

if True:
    max_iters = 100
    trainer = ScipyTrainer(m, 'L-BFGS-B')
    lc, _  = trainer.train(None, max_iters, callback=progress_bar_callback(max_iters))
    plt.plot(lc)
    plt.show()

print(m.get_objective())

m.print()

#print(m.get_objective())
#breakpoint()

# Predict
XS = X

if False:
    pred_mu, pred_var = m.predict_f(XS, filter_only=True, diagonal=False)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)
    pred_var = np.diagonal(pred_var, axis1=1, axis2=2)

    pred_mu = pred_mu[:, 0]
    pred_var = pred_var[:, 0]
else:
    pred_mu, pred_var = m.predict_f(XS, filter_only=False, diagonal=True)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)



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

