""" Gaussian Process Regression Computed through a State Space Representation"""

import sys
sys.path.append('../')

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
from stgp.trainers import ScipyTrainer, GradDescentTrainer
from stgp.trainers.standard import ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32, ScaledMatern32, ScaledMatern52, ScaledMatern72
from stgp.kernels.bias import ConstantKernel
from stgp.data import TemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent

stgp.settings.verbose = True

import matplotlib.pyplot as plt

np.random.seed(0)

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

Y = Y+10

# Construct Model
data = TemporalData(X, Y)
lik = ReshapedGaussian(Gaussian(variance=1.0), num_blocks=data.Nt, block_size=1)
#kern = ScaledMatern32(input_dim=1, lengthscales=[1.0], variance=1.0)
#kern = Matern32(input_dim=1, lengthscales=[1.0])
kern = ConstantKernel(variance=1.0)+Matern32(input_dim=1, lengthscales=[1.0])

latent_gp = GP(
    sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
    kernel = kern,
    prior = True
)
prior = LTI_SDE(Independent([latent_gp])) 

# try different Kalman filter and smoothers
#   (default) filter_type='sequential'
#   filter_type='parallel'
#   filter_type='square_root_svm' 
m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential', filter_type='sequential')


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
pred_mu, pred_var = m.predict_y(XS,  diagonal=True, squeeze=True)

# Plot
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

