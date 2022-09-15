import sys
sys.path.append('../')

import jax
from jax.config import config as jax_config
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
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

# Generate data
Q = 3
P = 3
N = 50

XS, X, Y = multi_output_timeseries(P, N, 500, seed=0)

X_test = np.copy(X)
Y_test = np.copy(Y)

# Remove section of Y for testing
Y[20:35, 1] = np.NaN

print(f'X: {X.shape}, Y: {Y.shape}')

if False:
    for p in range(P):
        plt.plot(X[:, 0], Y[:, p])
    plt.show()

# Construct Model
Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

# Construct Latent GPs

latent_W_gps = [
    [
        stgp.models.GP(sparsity=stgp.sparsity.NoSparsity(X), kernel=RBF(lengthscales=[0.1])) 
        for q in range(Q)
    ]
    for p in range(P)
] 

latent_f_gps = [
    stgp.models.GP(sparsity=stgp.sparsity.NoSparsity(X), kernel=RBF(lengthscales=[0.1])) for q in range(Q)
] 
prior = stgp.transforms.multi_output.GPRN(latent_W_gps, latent_f_gps, output_dim = P)

m = stgp.models.GP(
    data=Data(X, Y), 
    likelihood=[Gaussian(), Gaussian(), Gaussian()],
    inference='Variational',
    prior=prior,
    ell_samples = 10,
    prediction_samples = 1000
)

print(m.confidence_intervals(X))
print(m.get_objective())
print(m.predict_f(X))
print('NLPD: ', m.nlpd(X, Y))

if True:
    max_iters = 100

    ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()

    #trainer = ScipyTrainer(m, 'L-BFGS-B')
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    ng_trainer.train(0.01, 10)
    for i in trange(max_iters):
        trainer.train(0.01, 1)
        ng_trainer.train(0.1, 1)

print('NLPD: ', m.nlpd(X, Y))

pred_mu, pred_var = m.predict_y(XS, diagonal=True, output_first=True, squeeze=True)

breakpoint()

fig, axes = plt.subplots(P, 1, sharex=True)

for p in range(P):

    axes[p].fill_between(
        np.squeeze(XS), 
        np.squeeze(pred_mu[p] - 1.96*np.sqrt(pred_var[p])), 
        np.squeeze(pred_mu[p] + 1.96*np.sqrt(pred_var[p])), 
        facecolor=colors.LINE_COL, 
        alpha=0.3
    )
    axes[p].plot(XS, pred_mu[p], color=colors.LINE_COL, label='GP Fit')
    axes[p].scatter(X_test, Y_test[:, p], color='black', label='Testing Data')
    axes[p].scatter(X, Y[:, p], color='grey', label='Training Data')

    axes[p].legend()
plt.show()





