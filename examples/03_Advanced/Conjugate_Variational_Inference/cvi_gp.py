""" Conjugate Variational Gaussian Process Regression on a temporal dataset"""
import sys
sys.path.append('../')
sys.path.append('../../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, Matern52, ScaledMatern52
from stgp.likelihood import Gaussian, ProductLikelihood, GaussianProductLikelihood
from stgp.data import Data, TemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import MeanFieldConjugateGaussian, ConjugateGaussian

from tqdm import trange

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

print(f'X: {X.shape}, Y: {Y.shape}')

def cvi_gp():
    """ CVI-GP with a standard GP surrogate model parameterised using moment parameterisation.  """
    # Construct Model
    Q = 1
    sparsity = stgp.sparsity.NoSparsity(Z = X)

    kern = ScaleKernel(Matern52(input_dim=1, lengthscales=[0.1]))
    latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True)]

    data = Data(X, Y)
    m = GP(
        data = data,
        prior = Independent(latent_gps),
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior = MeanFieldConjugateGaussian([
            ConjugateGaussian(
                X=sparsity,
                num_blocks = data.N,
                block_size=1,
                num_latents=1,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    data=Data(X.X, Y), # Data should already be in the correct format
                    prior=Independent([latent_gps[q]]), 
                    likelihood=[likelihood] 
                )  
            )
            for q in range(Q)
        ]),
        inference='Variational'
    )
    return m

def cvi_sde_gp(parallel=False):
    """ CVI-GP with a state-space GP surrogate model parameterised using moment parameterisation.  """
    # Construct Model
    Q = 1
    sparsity = stgp.sparsity.NoSparsity(Z = X)

    kern = ScaledMatern52(input_dim=1, lengthscales=[0.1], variance=1.0)
    latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True)]

    data = Data(X, Y)
    m = GP(
        data = data,
        prior = Independent(latent_gps),
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior = MeanFieldConjugateGaussian([
            ConjugateGaussian(
                X=sparsity,
                num_blocks = data.N,
                block_size=1,
                num_latents=1,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    data=TemporalData(X.X, Y, sort=False), # Data should already be in the correct format
                    prior=LTI_SDE(Independent([latent_gps[q]])), 
                    likelihood=[likelihood],
                    inference='Sequential',
                    parallel=parallel
                )  
            )
            for q in range(Q)
        ]),
        inference='Variational'
    )
    return m

models = {
    'cvi_gp': cvi_gp(),
    'cvi_sde_gp_seq': cvi_sde_gp(parallel=False),
    'cvi_sde_gp_parallel': cvi_sde_gp(parallel=True),
}

if True:
    for k, m in models.items():
        ng_trainer = NatGradTrainer(m)
        ng_trainer.train(1.0, 1)

N_models = len(models)

fig, axes = plt.subplots(N_models, 1, squeeze=False)

for i, key   in enumerate(models):
    m = models[key]

    ax_i = axes[i][0]

    pred_mu, pred_var = m.predict_f(XS)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)

    ax_i.fill_between(
        np.squeeze(XS), 
        np.squeeze(pred_mu - 1.96*np.sqrt(pred_var)), 
        np.squeeze(pred_mu + 1.96*np.sqrt(pred_var)), 
        facecolor=colors.LINE_COL, 
        alpha=0.3
    )
    ax_i.plot(XS, pred_mu, color=colors.LINE_COL, label='GP Fit')
    ax_i.scatter(X, Y, color='black', label='Training Data')
    ax_i.set_title(key)
    ax_i.legend()
plt.show()
