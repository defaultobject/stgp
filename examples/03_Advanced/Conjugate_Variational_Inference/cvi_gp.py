""" Conjugate Variational Gaussian Process Regression """
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
from stgp.kernels import RBF, ScaleKernel
from stgp.likelihood import Gaussian, ProductLikelihood, GaussianProductLikelihood
from stgp.data import Data, TemporalData
from stgp.transforms import Independent
from stgp.approximate_posteriors import MeanFieldConjugateGaussian, ConjugateGaussian

from tqdm import trange

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

print(f'X: {X.shape}, Y: {Y.shape}')

# Construct Model
Q = 1
sparsity = stgp.sparsity.NoSparsity(Z = X)

kern = ScaleKernel(RBF(input_dim=1, lengthscales=[0.1]))
latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True)]

data = Data(X, Y)
m = GP(
    data = data,
    prior = Independent(latent_gps),
    likelihood = GaussianProductLikelihood([Gaussian()]),
    approximate_posterior = MeanFieldConjugateGaussian([
        ConjugateGaussian(
            X=sparsity,
            block_size=1,
            surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                data=Data(X.X, Y), # Data should already be in the correct format
                prior=Independent([latent_gps[q]]), 
                likelihood=[likelihood[q]]
            )  
        )
        for q in range(Q)
    ]),
    inference='Variational'
)

# Train
if True:
    max_iters = 500

    ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()

    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    lc, _ = ng_trainer.train(1.0, 1)
    lc_arr = [lc]
    for i in trange(max_iters):
        trainer.train(0.01, 1)
        lc_i, _ = ng_trainer.train(1.0, 1)
        lc_arr.append(lc_i)

    plt.plot(lc_arr)
    plt.show()

pred_mu, pred_var = m.predict_y(XS)

plt.fill_between(
    np.squeeze(XS), 
    np.squeeze(pred_mu - 1.96*np.sqrt(pred_var)), 
    np.squeeze(pred_mu + 1.96*np.sqrt(pred_var)), 
    facecolor=colors.LINE_COL, 
    alpha=0.3
)
plt.plot(XS, pred_mu, color=colors.LINE_COL, label='GP Fit')
plt.scatter(X, Y, color='black', label='Training Data')
plt.legend()
plt.show()


