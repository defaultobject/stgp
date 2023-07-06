""" Gaussian Process Regression Computed through a State Space Representation"""

import sys
sys.path.append('../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np
from jax import make_jaxpr

from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors

import stgp
from stgp import settings
from stgp.models import GP
from stgp.trainers import ScipyTrainer, GradDescentTrainer
from stgp.trainers.standard import ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF 
from stgp.data import SpatioTemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent

import matplotlib.pyplot as plt
from timeit import default_timer as timer

settings.jitter = 1e-7
#stgp.settings.linear_solver = stgp.settings.SolveType.CG
#stgp.settings.linear_solver = stgp.settings.SolveType.EXACT
stgp.settings.linear_solver = stgp.settings.SolveType.CHOLESKY

# Construct Data
XS, X, Y = single_output_spatial_data(100, 200, 200, 200, seed=0)

Y  = Y + X[:, 0][:, None] + X[:, 1][:, None]

# Construct Model
data = SpatioTemporalData(X=X, Y=Y, sort=True)
lik = ReshapedGaussian(Gaussian(), num_blocks=data.Nt, block_size=data.Ns)
kern= SpatioTemporalSeperableKernel(
    Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
    RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])
)

latent_gp = GP(
    sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
    kernel = kern,
    prior = True
)

prior = LTI_SDE(Independent([latent_gp])) 

parallel = False

print(f'parallel: {parallel}')

#m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential', filter_type='square_root_svm')
m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential')

#print(m.get_objective())
#breakpoint()

#m.predict_f(X)
#breakpoint()

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    print('starting timer')
    start = timer()
    fn = objax.Jit(m.get_objective, m.vars())
    print(fn())
    end = timer()
    print('end')
    print(end - start)

breakpoint()

#print(make_jaxpr(m.get_objective)())

# Train
if True:
    max_iters = 100
    if False:
        trainer = ScipyTrainer(m, 'L-BFGS-B')
        trainer.train(None, max_iters, callback=progress_bar_callback(max_iters))
    else:
        trainer = ADAM(m)
        trainer.train(1e-2, max_iters, callback=progress_bar_callback(max_iters))

    print(m.get_objective())

# Predict
pred_mu, pred_var = m.predict_f(XS)

print(pred_mu)

# Plot
fig, axes = plt.subplots(1, 2)

axes[0].set_title('Mean')
axes[0].imshow(pred_mu.reshape(200, 200))

axes[1].set_title('Variance')
axes[1].imshow(pred_var.reshape(200, 200))

plt.show()
