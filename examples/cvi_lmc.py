"""
This files show how to construct low level multi-task variational models in the
    1) temporal and spatio-temporal setting (--time, --st)
    2) mean-field and dense posterior setting (--mf, --fp)
    3) batch / sde CVI model with no sparsity (--batch, --sde) (--no-Z)
    4) sde CVI model with spatial sparsity (--spatial-Z)
"""
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32, SpatioTemporalSeperableKernel
from legogp.likelihood import Gaussian, BlockDiagonalGaussian, ReshapedBlockDiagonalGaussian
from legogp.data import Data, TemporalData, MultiOutputTemporalData, get_sequential_data_obj, SpatioTemporalData, DataReshape
from legogp.sparsity import NoSparsity, StackedNoSparsity, SpatialSparsity
from legogp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from legogp.transforms import DataLatentPermutation , Independent

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

from data_zoo import multi_output_spatial_data, multi_output_timeseries

import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--time', action='store_true')
parser.add_argument( '--st', action='store_true')
parser.add_argument( '--cvi', action='store_true')
parser.add_argument( '--vi', action='store_true')
parser.add_argument( '--fp', action='store_true')
parser.add_argument( '--mf', action='store_true')
parser.add_argument( '--no-Z', action='store_true')
parser.add_argument( '--dense-Z', action='store_true')
parser.add_argument( '--spatial-Z', action='store_true')
parser.add_argument( '--sde', action='store_true')
parser.add_argument( '--batch', action='store_true')

cmd_args = vars(parser.parse_args())

# Helper functions for plotting

def plot_data(X, Y, Nt, Ns, cmd_args):
    """ Helper function plotting spatio-temporal and timeseries data """
    P = Y.shape[-1]

    if cmd_args['st']:
        # ST data plotting
        fig, axes = plt.subplots(1, P)
        N = Y.shape[0]
        for p in range(P):
            axes[p].imshow(
                Y[:, p].reshape(Nt, Ns)
            )

    else:
        # Timeseries data plotting
        fig, ax = plt.subplots(1, 1)
        N = Y.shape[0]
        for p in range(P):
            ax.scatter(X, Y[:, p])

    plt.show()

def plot_res(mu, var, Nt, Ns, cmd_args):
    P = mu.shape[0]

    if cmd_args['st']:
        fig, axes = plt.subplots(1, P)

        N = Y.shape[0]
        for p in range(P):
            axes[p].imshow(
                mu[p].reshape(Nt, Ns)
            )
    else:
        # Timeseries data plotting
        fig, ax = plt.subplots(1, 1)
        N = Y.shape[0]
        for p in range(P):
            ax.fill_between(
                np.squeeze(XS), 
                np.squeeze(mu[p] - 2*np.sqrt(var[p])), 
                np.squeeze(mu[p] + 2*np.sqrt(var[p])), 
                alpha=0.4
            )
            ax.plot(XS, mu[p])
            ax.scatter(X, Y[:, p])


    plt.show()

# Data Generation
Q = 3
P = 3

if cmd_args['st']:
    print('Spatio-temporal dataset')
    Nt = 10
    Ns = 10

    Nts = 500
    Nss = 500

    XS, X, Y = multi_output_spatial_data(P, Nt, Ns, Nts, Nss, seed=0)

elif cmd_args['time']:
    print('Temporal dataset')
    Nt = 200
    Nts = 1000
    Ns = None
    Nss = None
    XS, X, Y = multi_output_timeseries(P, Nt, Nts, seed=0)
else:
    raise NotImplementedError()

if False :
    plot_data(X, Y, Nt, Ns, cmd_args)

# Setup Data format
if cmd_args['st']:
    if cmd_args['spatial_Z']:
        #  In the spatial sparsity setting we require kronecker structure on the surrogate model  
        st_data = data
        data = SpatioTemporalData(X=X, Y=Y, sort=True)
    else:
        data = Data(X=X, Y=Y)
        st_data = SpatioTemporalData(X=X, Y=Y, sort=True)

    input_data = st_data._X

elif cmd_args['time']:
    if cmd_args['sde']:
        pass
    else:
        data = Data(X=X, Y=Y)

    input_data = data._X

# Setup Sparsity
if cmd_args['no_Z']:
    print('No Z')
    if cmd_args['batch']:
        Z = [NoSparsity(X) for q in range(Q)]
        Z_all = StackedNoSparsity(Z)
    else:
        Z = [NoSparsity(Z_ref = input_data) for q in range(Q)]
        Z_all = StackedNoSparsity(Z)

elif cmd_args['spatial_Z']:
    M = 5
    Z_space = np.linspace(-1, 1, M)[:, None]
    Z = [SpatialSparsity(input_data.X_time, Z_space) for q in range(Q)]
    Z_all = None

elif cmd_args['dense_Z']:
    print('Dense Z')
    raise NotImplementedError()
else:
    raise NotImplementedError()


# Construct Latent GPs
if cmd_args['st']:
    latent_kernels = [
        SpatioTemporalSeperableKernel(
            Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
            Matern32(input_dim=1, lengthscales=[0.1], active_dims=[1])
        )
        for q in range(Q)
    ]
else:
    latent_kernels = [
        Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0])
        for q in range(Q)
    ]

latent_gps = [
    lego.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
] 

# Construct LMC Prior
prior = lego.transforms.multi_output.LMC(latent_gps, output_dim = P)

# Construct Approximate Posterior
if cmd_args['mf']:
    print('Mean Field Approximate Posterior')
    if cmd_args['no_Z']:
        block_size = 1
    elif cmd_args['spatial_Z']:
        block_size = M
    else:
        raise RuntimeError()

    if cmd_args['sde']:
        if cmd_args['no_Z']:
            q_cvi = MeanFieldConjugateGaussian([
                ConjugateGaussian(
                    X=Z[q],
                    block_size=block_size,
                    surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                        data=DataReshape(SpatioTemporalData(X=X.raw_Z, Y=Y, sort=False), new_shape=[st_data.Nt, st_data.Ns, 1]), # Data should already be in the correct format
                        prior=Independent([latent_gps[q]]), 
                        likelihood=ReshapedBlockDiagonalGaussian(likelihood[0], Nt, Ns),
                        inference='Sequential'
                    ) # batch gp surrogate model 
                )
                for q in range(Q)
            ])
        elif cmd_args['spatial_Z']:
            q_cvi = MeanFieldConjugateGaussian([
                ConjugateGaussian(
                    X=Z[q],
                    block_size=block_size,
                    surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                        data=SpatioTemporalData(X=X.raw_Z, Y=Y, sort=False), # Data should already be in the correct format
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
if cmd_args['spatial_Z']:
    data = DataReshape(data, new_shape=[data.N, P])

m = lego.models.GP(
    data=data,
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
    natgrad_trainer.train([1.0, 1.0], 1)
    #natgrad_trainer.train([0.01, 0.1], 10)
    #natgrad_trainer.train([0.1, 0.1], 5)

    print('OBJ after NG: ', m.get_objective())

# Predict
pred_mu, pred_var = m.predict_y(XS, diagonal=True)

plot_data(X, Y, Nt, Ns, cmd_args)
plot_res(pred_mu, pred_var, Nts, Nss, cmd_args)
