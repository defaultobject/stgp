import sys
sys.path.append('../../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import multi_output_spatial_data
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF 
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data, SpatioTemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian
from stgp.transforms import One2One
from stgp.computation.parameter_transforms import identity
import stgp
from stgp.models import GP
from stgp.sparsity import NoSparsity, SpatialSparsity, StackedSparsity, StackedNoSparsity

from tqdm import trange

import matplotlib.pyplot as plt

# Generate data
Q = 3
P = 3
N = 10

XS, X, Y = multi_output_spatial_data(P, N, N, 100, 100, seed=0)

if False:
    fig, axes = plt.subplots(1, P)
    for i in range(P):
        axes[i].imshow(Y[:, i].reshape(10, 10))
    plt.show()

print(f'XS: {XS.shape}, X: {X.shape}, Y: {Y.shape}')


st_data = SpatioTemporalData(X=X, Y=Y)

# Construct Model
#Z = [NoSparsity(Z_ref = st_data._X) for q in range(Q)]
Z = [SpatialSparsity(st_data.X_time, st_data.X_space[:5]) for q in range(Q)]
Z_all = StackedSparsity(Z)

# Construct Latent GPs
latent_kernels = [
    SpatioTemporalSeperableKernel(
        Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
        RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])
    )
    for q in range(Q)
]
latent_gps = [
    stgp.models.GP(sparsity=Z[0], kernel=latent_kernels[q]) for q in range(Q)
] 
np.random.seed(0)
W =  np.random.randn(P, Q)

prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim = P, input_dim=2, W = W)

# Construct Full Gaussian Approximate Posterior

Mt = Z[0].raw_Z.Nt
Ms = Z[0].raw_Z.Ns

q = FullConjugateGaussian(
    X = Z[0], # for state-space models we require the same Z across all latents
    num_latents=Q*Z[0].raw_Z.Ns,
    block_size=Q*Z[0].raw_Z.Ns,
    num_blocks = st_data.Nt,
    surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
        data = SpatioTemporalData(X=X.raw_Z, Y=np.reshape(Y, [Mt, Ms, Q]), sort=False), # we need gradients Y so set to be trainable
        likelihood=likelihood, 
        prior=LTI_SDE(Independent(latent_gps)),
        inference='Sequential',
        full_state_observed = False
    )
)
m = stgp.models.GP(
    data=st_data, 
    likelihood=[Gaussian(0.1), Gaussian(0.1), Gaussian(0.1)],
    inference='Variational',
    prior=prior,
    approximate_posterior = q  
)

print(m.get_objective())

breakpoint()

print('NLPD: ', m.nlpd(X, Y, num_samples=10000))


pred_mu, pred_var = m.predict_f(XS)

if True:
    max_iters = 200

    #ng_trainer = NatGradTrainer(m)
    #ng_trainer = NatGradTrainer(m, enforce_psd_type='gauss_newton', prediction_samples=100)
    ng_trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton')
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






