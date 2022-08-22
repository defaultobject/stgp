""" Single GP regression """
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp import settings
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D, SecondOrderDerivativeKernel_1D, SecondOrderDerivativeKernel_1D
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms import LinearTransform, OutputMap, MultiOutput, DataLatentPermutation, Independent
from stgp.core.model_types import get_model_type
from stgp.transforms.pdes import DifferentialOperatorJoint, HeatEquation2D, Pendulum1D
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior

from data_zoo import single_output_spatial_data

import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import stdata
import stdata as st
from stdata.plots import grid_to_matrix
from stdata.grids import create_spatial_temporal_grid, create_spatial_grid

from tqdm import tqdm, trange

# Fix randomness
np.random.seed(0)

Nt = 100
T = np.linspace(0, 50, Nt)
X = T[:, None]
XS = np.linspace(0, 50, 1000)[:, None]

Y = (np.sin(T) + 0.01*np.random.randn(T.shape[0]))[:, None]

nan_idx = (X>20)
Y[nan_idx] = np.NaN

base_kernel_1d = RBF(input_dim = 1, lengthscales = [1.0])
kern = SecondOrderDerivativeKernel_1D(base_kernel_1d)

Kxx = kern.K(X, X)

if False:
    # check out samples
    latent = np.random.multivariate_normal(np.zeros(Kxx.shape[0]), Kxx)

    # there are three outputs
    latent_t = latent[:Nt]
    latent_dt = latent[Nt:Nt*2]
    approx_dt = np.gradient(latent_t, T)
    latent_dt2 = latent[Nt*2:]
    approx_dt2 = np.gradient(latent_dt, T)

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(latent_t, color='black')

    axes[1].plot(latent_dt, color='black')
    axes[1].plot(approx_dt, color='red', linestyle=(0, (5, 10)))

    axes[2].plot(latent_dt2, color='black')
    axes[2].plot(approx_dt2, color='red', linestyle=(0, (5, 10)))
    plt.show()

if False:
    plt.plot(X, Y)
    plt.show()

Y = np.hstack([Y, np.zeros_like(Y)])

#base_kernel_1d.lengthscale_param.fix()
settings.ng_jitter = 1e-5

if True:
    diff_op_prior = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel_1d
        ),
        kern
    )
    prior_output_1, prior_output_2 = OutputMap(
        diff_op_prior, 
        [[0], [0, 1, 2]], 
    )

    pde_output = Pendulum1D(prior_output_2)

    prior = MultiOutput([
        prior_output_1,
        pde_output
    ])
    lik_arr = [Gaussian(variance=0.01), Gaussian(variance=0.001)]

    lik_arr[0].variance_param.fix() 
    lik_arr[1].variance_param.fix() 

    data = stgp.data.Data(X, Y)

    # Defined in data-latent format?
    q = FullGaussianApproximatePosterior(dim = X.shape[0] * prior.base_prior.output_dim)
elif False:
    prior = Independent([GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_kernel_1d
    )])

    lik_arr = [Gaussian(variance=0.01)]

    data = stgp.data.Data(X, Y[:, 0][:, None])

    q = None
else:
    diff_op_prior = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel_1d
        ),
        kern
    )
    prior = OutputMap(
        diff_op_prior, 
        [[0]], 
    )


    lik_arr = [Gaussian(variance=0.01)]

    data = stgp.data.Data(X, Y[:, 0][:, None])

    # Defined in data-latent format?
    q = FullGaussianApproximatePosterior(dim = X.shape[0] * prior.base_prior.output_dim)



# Create Model
m = stgp.models.GP(
    data = data,
    prior = prior,
    likelihood = lik_arr,
    inference='Variational',
    approximate_posterior = q,
    ell_samples = 100,
    prediction_samples = 1000,
    whiten=True
)

m.print()

print(m.get_objective())
if False:
    # Train
    epochs = 500

    callback = progress_bar_callback(epochs)

    learning_curve, training_time = GradDescentTrainer(
        m, 
        objax.optimizer.Adam,
    ).train(
        0.01,
        epochs,
        callback = callback
    )

    # Plot learning curve
    if True:
        print(learning_curve[0], learning_curve[-1])
        plt.plot(learning_curve)
        plt.yscale('log')
        plt.show()

if False:
    #ng_trainer = NatGradTrainer(m, enforce_psd_type='retraction')
    ng_trainer = NatGradTrainer(m, enforce_psd_type='retraction')
    #ng_trainer = NatGradTrainer(m)
    #ng_trainer.train(0.01, 20) 
    lc_arr, _ = ng_trainer.train(0.01, 10) 
    lc_arr, _ = ng_trainer.train(0.1, 100) 
    plt.plot(lc_arr)
    plt.show()
    #ng_trainer.train(1.0, 1) 

    #ng_trainer.train(0.5, 10) 
    #ng_trainer.train(1.0, 1) 

if True:
    epochs = 200

    ng_trainer = NatGradTrainer(m,  enforce_psd_type='retraction')
    gd_trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    ng_trainer.train(0.01, 20) 
    learning_curve = []
    for i in trange(epochs):
        gd_trainer.train(0.001, 1)
        lc_i, _ = ng_trainer.train(0.01, 1) 
        learning_curve.append(lc_i)

    # Plot learning curve
    if True:
        print(learning_curve[0], learning_curve[-1])
        plt.plot(learning_curve)
        plt.show()


print(m.get_objective())

m.print()

pred_mu, pred_var = m.predict_f(XS, squeeze=False)

plt.scatter(T, Y[:, 0])
plt.plot(XS[:, 0], pred_mu[0])
plt.show()


