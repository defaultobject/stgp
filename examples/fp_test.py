import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax

import legogp as lego
from legogp import settings
from legogp.trainers import ScipyTrainer, NatGradTrainer, GradDescentTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32, ScaleKernel
from legogp.likelihood import Gaussian, ReshapedGaussian
from legogp.data import TemporalData, Data, MultiOutputTemporalData
from legogp.models import GP
from legogp.sparsity import NoSparsity, StackedNoSparsity
from legogp.transforms import Independent, DataLatentPermutation
from legogp.transforms.multi_output import LMC, GPRN_DRD
from legogp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from legogp.approximate_posteriors import FullGaussianApproximatePosterior

import matplotlib.pyplot as plt

import numpy as np

from data_zoo import multi_output_timeseries

np.random.seed(0)

P = 2
Q = 2

XS, X, Y = multi_output_timeseries(P, 100, 1000, seed=0)

config = {
    'ls': [0.1, 0.1, 0.1],
    'lik_var': [0.1, 0.1, 0.1],
    'epochs': 100,
    'lr': 0.01,
    'P': P,
    'Q': Q
}

res_time = {}

def dense_sde_cvi_lmc(config, XS, X, Y):
    settings.ng_jitter = 1e-8
    data = Data(X, Y)
    
    Z = [NoSparsity(Z_ref=data._X) for q in range(Q)]
    Z_all = StackedNoSparsity(Z)
    
    latents = [
        lego.models.GP(
            sparsity = Z[q], 
            kernel = Matern32(input_dim=1, lengthscales=[config['ls'][q]], active_dims=[0]),
            latent = True
        ) 
        for q in range(Q)
    ]
    
    prior = GPRN_DRD([latents[0]], latents, output_dim = P)
    
    lik = [Gaussian(config['lik_var'][p]) for p in range(P)]
    
    approximate_posterior = FullGaussianApproximatePosterior(
        dim = Z[0].shape[0]*prior.num_latents
    )
    prior = DataLatentPermutation(prior)

    m = GP(
        data = data,
        prior = prior,
        likelihood=lik,
        approximate_posterior=approximate_posterior,
        inference='Variational',
        prediction_samples=100,
        ell_samples=100
    )
    print(m.get_objective())

    m.predict_y(X)
    
    # Natgrad step
    natgrad_trainer = NatGradTrainer(m, schedule='constant')
    
    # Do not use adam for the approx posterios
    all_vars = list(m.vars().keys())
    m_name = [a for a in all_vars if a.endswith('._m(Parameter).raw_var')]
    s_chol_name = [a for a in all_vars if a.endswith('._S_chol(Parameter).raw_var')]
    approx_posterior_vars = m_name + s_chol_name
            
    grad_step = GradDescentTrainer(m, objax.optimizer.Adam, hold_vars = approx_posterior_vars)
    
    learning_curve = []
    for i in range(config['epochs']):
        obj_val, _ = natgrad_trainer.train(1.0, epochs=1)
        learning_curve.append(obj_val[0])
        grad_step.train(config['lr'], epochs=1)

    pred_mu, pred_var = m.predict_y(XS)

    
    return {'m': m, 'lc': learning_curve, 'pred_mu': pred_mu, 'pred_var': pred_var}
    
res_time['dense_sde_cvi'] = dense_sde_cvi_lmc(config, XS, X, Y)

