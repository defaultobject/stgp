import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32
from legogp.likelihood import Gaussian
from legogp.data import Data, TemporalData
from legogp.sparsity import NoSparsity
from legogp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian

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

# Generate data
Q = 2
P = 2

XS, X, Y = multi_output_timeseries(P, 200, 1000, seed=0)

# TODO: define kernels and sparsity across all latents

Z = NoSparsity(X)

latent_f_kernels = [Matern32(lengthscales=[0.01]) for q in range(Q)]
latent_W_kernels = [Matern32(lengthscales=[1.0]), Matern32(lengthscales=[1.0]), Matern32(lengthscales=[1.0])]

latent_kernels = latent_f_kernels + latent_W_kernels

# Construct Prior
f_latents = [
    lego.models.GP(sparsity=Z, kernel=latent_f_kernels[q], latent=True) for q in range(Q)
]

W_latents = [
    lego.models.GP(sparsity=Z, kernel=latent_W_kernels[0], latent=True)
]

prior = lego.transforms.multi_output.GPRN_LDL(W_latents, f_latents, input_dim = Q, output_dim = P)


# Construct Approximate Posterior
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
    for q in range(prior.num_latents)
])

# Create Model
m = lego.models.GP(
    data=Data(X, Y),
    likelihood = [Gaussian(variance=0.1) for p in range(P)],
    prior=prior,
    approximate_posterior=q_cvi,
    inference='Variational',
    ell_samples=1000,
    prediction_samples=1000
)

print(m.get_objective())

if True:
    lego.settings.ng_jitter = 1e-5

    # NatGrad trainer
    natgrad_trainer = NatGradTrainer(m, schedule='linear')
    natgrad_trainer.train([0.01, 0.01], 10)
    natgrad_trainer.train([0.01, 0.1], 10)
    #natgrad_trainer.train([0.1, 0.1], 1)

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
