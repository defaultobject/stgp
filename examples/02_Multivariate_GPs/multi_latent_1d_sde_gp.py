""" Multi-latent State-Space Model constructed by stacking the states together """
import sys
sys.path.append('../')

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import multi_output_timeseries
from example_utils import colors

import stgp
from stgp.models import GP
from stgp.trainers import ScipyTrainer, GradDescentTrainer
from stgp.trainers.standard import ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32, ScaledMatern32, ScaledMatern52
from stgp.data import TemporalData, MultiOutputTemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian, GaussianProductLikelihood
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent

import matplotlib.pyplot as plt

# Generate data
Q = 3
P = 3
N = 50

XS, X, Y = multi_output_timeseries(P, N, 10, seed=0)

if False:
    for p in range(P):
        plt.plot(X[:, 0], Y[:, p])
    plt.show()

# independent GP on each task

# Construct Model
data = MultiOutputTemporalData(X, Y)
lik = ReshapedGaussian(
    GaussianProductLikelihood([Gaussian(variance=1.0) for q in range(Q)]),
    num_blocks=data.Nt, 
    block_size=Q
)
prior = LTI_SDE(Independent(
    [
        GP(
            sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
            kernel = ScaledMatern52(input_dim=1, lengthscales=[1.0], variance=1.0),
            prior = True
        )
        for q in range(Q)
    ]
)) 

# try different Kalman filter and smoothers
#   (default) filter_type='sequential'
#   filter_type='parallel'
#   filter_type='square_root_svm' 
m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential')

if True:
    max_iters = 100
    trainer = ScipyTrainer(m, 'L-BFGS-B')
    lc, _  = trainer.train(None, max_iters, callback=progress_bar_callback(max_iters))
    plt.plot(lc)
    plt.show()

print(m.get_objective())
pred_mu, pred_var = m.predict_y(XS, diagonal=True, squeeze=True)

pred_mu = pred_mu.T
pred_var = pred_var.T

fig, axes = plt.subplots(P, 1, sharex=True)

for p in range(P):

    axes[p].fill_between(
        np.squeeze(XS), 
        np.squeeze(pred_mu[p] - 1.96*np.sqrt(pred_var[p])), 
        np.squeeze(pred_mu[p] + 1.96*np.sqrt(pred_var[p])), 
        facecolor=colors.LINE_COL, 
        alpha=0.3
    )
    axes[p].plot(np.squeeze(XS), pred_mu[p], color=colors.LINE_COL, label='GP Fit')
    axes[p].scatter(np.squeeze(X), Y[:, p], color='grey', label='Training Data')

    axes[p].legend()
plt.show()


