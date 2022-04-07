import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax

import legogp as lego
from legogp import settings
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer, GradDescentTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32, ScaleKernel, RBF
from legogp.likelihood import Gaussian, ReshapedGaussian
from legogp.data import TemporalData, Data, MultiOutputTemporalData
from legogp.models import GP
from legogp.sparsity import NoSparsity, StackedNoSparsity, FullSparsity
from legogp.transforms import Independent, DataLatentPermutation
from legogp.transforms.multi_output import LMC
from legogp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from legogp.approximate_posteriors import FullGaussianApproximatePosterior

import matplotlib.pyplot as plt

import numpy as np

from data_zoo import single_output_timeseries

np.random.seed(0)


XS, X, Y = single_output_timeseries(100, 1000, seed=0)

config = {
    'ls': [0.1, 0.1, 0.1],
    'lik_var': [0.1, 0.1, 0.1],
    'epochs': 100,
    'lr': 0.01,
}

res_time = {}

def svgp(config, XS, X, Y):
    data = Data(X, Y)

    M = 3
    Z = np.linspace(0, 1, M)[:, None]
    
    Z = FullSparsity(Z=Z)
    
    latents = [
        lego.models.GP(
            sparsity=Z, 
            kernel=ScaleKernel(RBF(input_dim=1, lengthscales=[config['ls'][0]], active_dims=[0])),
            latent=True
        ) 
    ]
    
    prior = Independent(latents)
    
    lik = [Gaussian(config['lik_var'][0])]
    
    m = GP(
        data = data,
        prior = prior,
        likelihood=lik,
        inference='Variational',
        prediction_samples = 1000
    )

    print(m.get_objective())
    
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        config['lr'],
        config['epochs'],
        callback = None,
    )
    
    return {'m': m, 'lc': learning_curve}
    
res_time['batch'] = svgp(config, XS, X, Y)

m = res_time['batch']['m']

pred_mu, pred_var = m.predict_y(XS)

plt.fill_between(np.squeeze(XS), np.squeeze(pred_mu + 1.96 * np.sqrt(pred_var)), np.squeeze(pred_mu - 1.96 * np.sqrt(pred_var)), alpha = 0.4)
plt.plot(XS, pred_mu)
plt.scatter(X, Y)
plt.show()
