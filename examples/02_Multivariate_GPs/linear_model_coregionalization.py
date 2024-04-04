import sys
sys.path.append('../')

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
import objax
import numpy as np

from example_utils.data_zoo import multi_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF 
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data
from stgp.transforms import Independent
from tqdm import trange

from stgp.transforms import One2One
from stgp.transforms.basic import InvProbit
from stgp.computation.parameter_transforms import identity
import stgp
from stgp.models import GP

from tqdm import trange
import matplotlib.pyplot as plt

from stgp.metrics.nlpd import nlpd

# Generate data
Q = 3
P = 3
N = 50

XS, X, Y = multi_output_timeseries(P, N, 500, seed=0)

X_test = np.copy(X)
Y_test = np.copy(Y)

# Remove section of Y for testing
Y[25:35, 1] = np.NaN

print(f'X: {X.shape}, Y: {Y.shape}')

if False:
    for p in range(P):
        plt.plot(X[:, 0], Y[:, p])
    plt.show()

# Construct Model
Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

# Construct Latent GPs
latent_kernels = [RBF(lengthscales=[0.1]) for q in range(Q)]
latent_gps = [
    stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
] 
prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim = P)

m = stgp.models.GP(
    data=Data(X, Y), 
    likelihood=[Gaussian(0.1), Gaussian(0.1), Gaussian(0.1)],
    inference='Batch',
    prior=prior
)

#print('NLPD: ', nlpd(X, Y, m))

print(m.get_objective())

# Train
if False:
    max_iters = 200
    trainer = ScipyTrainer(m, 'L-BFGS-B')
    trainer.train(None, max_iters, callback=progress_bar_callback(max_iters))

else:
    max_iters = 500
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)
    trainer.train(0.01, max_iters, callback=progress_bar_callback(max_iters))

print(m.get_objective())
#print('NLPD: ', nlpd(X, Y, m))

pred_mu, pred_var = m.predict_f(XS)

pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)


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
    axes[p].scatter(np.squeeze(X_test), Y_test[:, p], color='black', label='Testing Data')
    axes[p].scatter(np.squeeze(X), Y[:, p], color='grey', label='Training Data')

    axes[p].legend()
plt.show()




