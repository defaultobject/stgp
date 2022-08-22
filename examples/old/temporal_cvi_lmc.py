import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp as lego
from stgp.trainers import SimpleTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32
from stgp.likelihood import Gaussian, BlockDiagonalGaussian
from stgp.data import Data, TemporalData, MultiOutputTemporalData, get_sequential_data_obj
from stgp.sparsity import NoSparsity, StackedNoSparsity
from stgp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from stgp.transforms import DataLatentPermutation , Independent

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
parser.add_argument( '--sde', action='store_true')
parser.add_argument( '--batch', action='store_true')

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
    lego.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
] 
prior = lego.transforms.multi_output.LMC(latent_gps, output_dim = P)

if True:
    sde_gp = lego.models.GP(
        data=get_sequential_data_obj(X, Y, sort=True), 
        likelihood=BlockDiagonalGaussian(block_size=Q, num_blocks=N),
        inference='Sequential',
        prior=prior.latent_obj
    )
    print(sde_gp.get_objective())
    sde_mu, sde_var = sde_gp.predict_f(XS)

    plt.plot(sde_mu)
    plt.show()
    breakpoint()

# Construct Approximate Posterior

if cmd_args['mf']:
    print('Mean Field Approximate Posterior')
    block_size = 1

    if cmd_args['sde']:
        q_cvi = MeanFieldConjugateGaussian([
            ConjugateGaussian(
                X=Z[q],
                block_size=block_size,
                surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                    data=TemporalData(X, Y, sort=False), # Data should already be in the correct format
                    prior=Independent([latent_gps[q]]), 
                    likelihood=likelihood[0],
                    inference='Sequential'
                ) # batch gp surrogate model 
            )
            for q in range(Q)
        ])
    elif cmd_args['batch']:
        q_cvi = MeanFieldConjugateGaussian([
            ConjugateGaussian(
                X=Z[q],
                block_size=block_size,
                surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                    data=Data(X, Y[..., 0]),
                    prior=Independent([latent_gps[q]]),
                    likelihood=likelihood[0],
                    inference='Batch'
                ) # batch gp surrogate model 
            )
            for q in range(Q)
        ])
    else:
        raise RuntimeError()

elif cmd_args['fp']:
    print('Full Posterior Approximate Posterior')
    block_size = Q

    # TODO: figure out X/Z here
    #Z_all = [Z for q in range(Q)]

    if cmd_args['batch']:
        q_cvi = FullConjugateGaussian(
            X = Z_all,
            num_latents=Q,
            block_size=block_size,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                data=Data(X, Y), 
                likelihood=likelihood, 
                prior=DataLatentPermutation(prior.latent_obj)
            )
        )
    elif cmd_args['sde']:
        # TODO: Z is confusing here
        q_cvi = FullConjugateGaussian(
            X = Z_all,
            num_latents=Q,
            block_size=block_size,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                data=MultiOutputTemporalData(X.sparsity_arr[0], Y[..., None], sort=False), 
                likelihood=likelihood, 
                prior=prior.latent_obj,
                inference='Sequential'
            )
        )

    else:
        raise RuntimeError()

else:
    raise RuntimeError()

# Create Model
m = lego.models.GP(
    data=Data(X, Y),
    likelihood = [Gaussian(variance=0.1) for p in range(P)],
    prior=prior,
    approximate_posterior=q_cvi,
    inference='Variational',
    ell_samples=100,
    prediction_samples=1000
)

print(m.get_objective())

if True:
    # NatGrad trainer
    natgrad_trainer = NatGradTrainer(m, schedule='linear')
    #natgrad_trainer.train([1.0, 1.0], 1)
    natgrad_trainer.train([0.01, 0.1], 10)
    natgrad_trainer.train([0.1, 0.1], 5)

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
