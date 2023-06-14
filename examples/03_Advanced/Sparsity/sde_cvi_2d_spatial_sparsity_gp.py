import sys
sys.path.append('../../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors
from stgp.trainers.standard import VB_NG_ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF 
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data, SpatioTemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian
from stgp.transforms import One2One
from stgp.computation.parameter_transforms import identity
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs_With_Mask
import stgp
from stgp.models import GP
from stgp.sparsity import NoSparsity, SpatialSparsity, StackedSparsity, StackedNoSparsity

from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

stgp.settings.jitter = 1e-5

# Construct Data
NS_nt = 100
NS_ns = 100
Q = 1
P = 1
XS, X, Y = single_output_spatial_data(10, 10, NS_nt, NS_ns, seed=0)

print(f'XS: {XS.shape}, X: {X.shape}, Y: {Y.shape}')

st_data = SpatioTemporalData(X=X, Y=Y)
st_data_xs = SpatioTemporalData(X=XS, Y=None)

print('spatial sparsity')
# Construct Model
# setup 10 inducing points
M = 3
Zs = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), M)[:, None]
Z = [SpatialSparsity(st_data.X_time, Zs, train=True) for q in range(Q)]

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

prior = stgp.transforms.Independent(latent_gps)

# Construct Full Gaussian Approximate Posterior
Mt = Z[0].raw_Z.Nt
Ms = Z[0].raw_Z.Ns


q = FullConjugateGaussian(
    X = Z[0], # for state-space models we require the same Z across all latents
    num_latents=Q,
    block_size=Q*Z[0].raw_Z.Ns,
    num_blocks = st_data.Nt,
    surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
        data = SpatioTemporalData(X=X.raw_Z, Y=np.reshape(Y, [Mt, Q, Ms]), sort=False), # we need gradients Y so set to be trainable, in time-latent-space format
        likelihood=likelihood, 
        prior=LTI_SDE_Full_State_Obs_With_Mask(Independent(latent_gps), keep_dims=[0]),
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

# Train
if True:
    print(m.get_objective())
    #trainer = VB_NG_ADAM(m, enforce_psd_type='laplace_gauss_newton')
    trainer = VB_NG_ADAM(m)
    trainer.ng_trainer.train(1.0, 1)
    print(m.get_objective())
    m.print()

    max_iters = 100
    lc, _ = trainer.train([0.01, 1.0], [max_iters, [1, 1]], callback=progress_bar_callback(max_iters))

    plt.plot(lc)
    plt.show()
    print(m.get_objective())

m.print()

pred_mu, pred_var = m.predict_y(XS, squeeze=False)

fig, axes = plt.subplots(1, 2, squeeze=False)
Nt = st_data.Nt
Ns = st_data.Ns

norm = Normalize(np.min(Y[:, 0]), np.max(Y[:, 0]))

axes[0][0].imshow(
    pred_mu.reshape(NS_nt, NS_ns), 
    extent=[np.min(XS[:, 0]), np.max(XS[:, 0]), np.min(XS[:, 1]), np.max(XS[:, 1])],
    origin='lower',
    norm=norm
)
axes[0][0].scatter(X[:, 1], X[:, 0], c=Y[:, 0], edgecolor='white', norm=norm)
plt.show()

breakpoint()


