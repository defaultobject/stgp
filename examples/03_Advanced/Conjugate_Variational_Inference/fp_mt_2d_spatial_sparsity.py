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
Q = 2
P = 2
N = 20
Nt = 100
Ns = 100

XS, X, Y = multi_output_spatial_data(P, N, N, Nt, Ns, seed=0)

Y = np.hstack([Y[:, 0][:, None] for p in range(P)])

if False:
    fig, axes = plt.subplots(1, P)
    for i in range(P):
        axes[i].imshow(Y[:, i].reshape(10, 10))
    plt.show()

print(f'XS: {XS.shape}, X: {X.shape}, Y: {Y.shape}')

st_data = SpatioTemporalData(X=X, Y=Y)
st_data_xs = SpatioTemporalData(X=XS, Y=None)

# Construct Model
#Z = [NoSparsity(Z_ref = st_data._X) for q in range(Q)]
Z = [SpatialSparsity(st_data.X_time, st_data.X_space[:10]) for q in range(Q)]
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

prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim = P, input_dim=Q, W = np.eye(P))
prior._W.fix()

# Construct Full Gaussian Approximate Posterior

Mt = Z[0].raw_Z.Nt
Ms = Z[0].raw_Z.Ns

q = FullConjugateGaussian(
    X = Z[0], # for state-space models we require the same Z across all latents
    num_latents=Q*Z[0].raw_Z.Ns,
    block_size=Q*Z[0].raw_Z.Ns,
    num_blocks = st_data.Nt,
    surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
        data = SpatioTemporalData(X=X.raw_Z, Y=np.reshape(Y, [Mt, Q, Ms]), sort=False), # we need gradients Y so set to be trainable, in time-latent-space format
        likelihood=likelihood, 
        prior=LTI_SDE(Independent(latent_gps)),
        inference='Sequential',
        full_state_observed = False
    )
)

m = stgp.models.GP(
    data=st_data, 
    likelihood=[Gaussian(0.1) for p in range(P)],
    inference='Variational',
    prior=prior,
    approximate_posterior = q,
    whiten=False
)

m.print()

if True:
    ng_trainer = NatGradTrainer(m, return_objective=False)
    m.approximate_posterior.fix()
    ng_trainer.train(1.0, 1)

if False:
    max_iters = 200

    ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    ll_arr, _ = ng_trainer.train(1.0, 1)
    ll_arr = [float(ll_arr)]

    print(m.get_objective())

    for i in trange(max_iters):
        trainer.train(0.01, 1)
        ll_i, _ = ng_trainer.train(1.0, 1)
        ll_arr.append(float(ll_i))

    print(m.get_objective())

    plt.plot(ll_arr)
    plt.show()

m.print()

pred_mu, pred_var = m.predict_f(XS)

if True:
    fig, axes = plt.subplots(2, P)
    for i in range(P):
        axes[0][i].imshow(pred_mu[:, i].reshape(Nt, Ns))
        axes[1][i].imshow(Y[:, i].reshape(N, N))
    plt.show()
