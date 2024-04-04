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
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF 
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data
from stgp.transforms import Independent
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
from stgp.transforms import One2One
from stgp.transforms.basic import InvProbit
from stgp.computation.parameter_transforms import identity
import stgp
from stgp.models import GP

from tqdm import trange

import matplotlib.pyplot as plt

# Generate data
Q = 3
P = 3
N = 50

XS, X, Y = multi_output_timeseries(P, N, 10, seed=0)

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
latent_kernels = [RBF(lengthscales=[0.1]) for q in range(Q)]
latent_gps = [
    stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
] 
np.random.seed(0)
W =  np.random.randn(P, Q)
print('W: ', W)

prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim = P, input_dim=2, W = W)

m = stgp.models.GP(
    data=Data(X, Y), 
    likelihood=[Gaussian(0.1), Gaussian(1.0), Gaussian(2.0)],
    inference='Variational',
    prior=prior,
    approximate_posterior = FullGaussianApproximatePosterior(dim = X.shape[0] * prior.base_prior.output_dim)
)


print('NLPD: ', m.nlpd(X, Y, num_samples=10000))


pred_mu, pred_var = m.predict_f(XS)

if True:
    max_iters = 500

    ng_trainer = NatGradTrainer(m)
    #ng_trainer = NatGradTrainer(m, enforce_psd_type='gauss_newton', prediction_samples=100)
    #ng_trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton')
    m.approximate_posterior.fix()

    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    ng_trainer.train(1.0, 1)
    if True:
        print(m.get_objective())

        for i in trange(max_iters):
            trainer.train(0.01, 1)
            ng_trainer.train(1.0, 1)

        print(m.get_objective())

GradDescentTrainer(m, objax.optimizer.Adam).train(0.01, 100)

print('NLPD: ', m.nlpd(X, Y, num_samples=10000))

pred_mu, pred_var = m.predict_y(XS, diagonal=True, output_first=True)

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





