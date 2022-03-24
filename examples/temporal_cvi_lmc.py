import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32
from legogp.likelihood import Gaussian, BlockDiagonalGaussian
from legogp.data import Data, TemporalData
from legogp.sparsity import NoSparsity, StackedNoSparsity
from legogp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from legogp.transforms import DataLatentPermutation 

import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import stdata as st
from stdata.plots import grid_to_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import stdata
import pandas as pd

from data_zoo import multi_output_timeseries

import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--cvi', action='store_true')
parser.add_argument( '--vi', action='store_true')
parser.add_argument( '--fp', action='store_true')
parser.add_argument( '--mf', action='store_true')
parser.add_argument( '--no-Z', action='store_true')
parser.add_argument( '--dense-Z', action='store_true')

cmd_args = vars(parser.parse_args())


# Generate data
Q = 3
P = 3
N = 50

XS, X, Y = multi_output_timeseries(P, N, 500, seed=0)

Z = [NoSparsity(X) for q in range(Q)]
Z_all = StackedNoSparsity(Z)

# Construct Latent GPs
latent_kernels = [Matern32(lengthscales=[0.1]) for q in range(Q)]
latent_gps = [
    lego.models.GP(sparsity=Z, kernel=latent_kernels[q]) for q in range(Q)
] 
prior = lego.transforms.multi_output.LMC(latent_gps, output_dim = P)

sde_gp = lego.models.GP(
    data=TemporalData(X, Y[..., None], sort=False), 
    likelihood=BlockDiagonalGaussian(block_size=Q, num_blocks=N),
    inference='Sequential'
)

print(sde_gp.get_objective())

breakpoint()
#TODO: fix the data sorting
sde_mu, sde_var = sde_gp.predict_f(XS)

breakpoint()

# Construct Approximate Posterior

if cmd_args['mf']:
    print('Mean Field Approximate Posterior')
    block_size = 1

    q_cvi = MeanFieldConjugateGaussian([
        ConjugateGaussian(
            X=Z,
            block_size=block_size,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                data=TemporalData(X, Y, sort=False), # Data should already be in the correct format
                kernel=latent_kernels[q], 
                likelihood=likelihood[0],
                inference='Sequential'
            ) # batch gp surrogate model 
        )
        for q in range(Q)
    ])
elif cmd_args['fp']:
    print('Full Posterior Approximate Posterior')
    block_size = Q

    # TODO: figure out X/Z here
    q = FullConjugateGaussian(
        X = Z_all,
        num_latents=Q,
        block_size=block_size,
        surrogate_model = lambda X, Y, likelihood:  lego.models.GP(data=Data(X, Y), likelihood=likelihood, prior=DataLatentPermutation(prior.latent_obj)), # use independent prior
    )

    breakpoint()

# Create Model
m = lego.models.GP(
    data=Data(X, Y),
    likelihood = [Gaussian(variance=0.1) for p in range(P)],
    prior=prior,
    approximate_posterior=q_cvi,
    inference='Variational'
)

print(m.get_objective())

if True:
    # NatGrad trainer
    natgrad_trainer = NatGradTrainer(m, schedule='linear')
    natgrad_trainer.train([1.0, 1.0], 1)

    print('OBJ after NG: ', m.get_objective())

# Predict
pred_mu, pred_var = m.predict_y(XS, diagonal=True)

# Plot results
fig = plt.figure(figsize=(10, 5))
ax = plt.gca()

for p in range(P):
    ax.fill_between(
        np.squeeze(XS), 
        np.squeeze(pred_mu[p] - 2*np.sqrt(pred_var[p])), 
        np.squeeze(pred_mu[p] + 2*np.sqrt(pred_var[p])), 
        alpha=0.4
    )
    ax.plot(XS, pred_mu[p])
    ax.scatter(X, Y[:, p])

plt.show()
