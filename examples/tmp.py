import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax



import legogp as lego
from legogp import settings
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer, GradDescentTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32
from legogp.likelihood import Gaussian, ReshapedGaussian
from legogp.data import TemporalData, Data
from legogp.models import GP
from legogp.sparsity import NoSparsity
from legogp.transforms import Independent
from legogp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian


import matplotlib.pyplot as plt

import numpy as np

from data_zoo import single_output_timeseries

# Fix randomness
np.random.seed(0)

XS, X, Y = single_output_timeseries(100, 1000, seed=0)

config = {
    'ls': 0.1,
    'lik_var': 0.1,
    'epochs': 100,
    'lr': 0.01
}

res_time = {}

def vi_timeseries_gp(config, XS, X, Y):
    settings.ng_jitter = 1e-7
    
    data = Data(X, Y)
    sparsity = NoSparsity(Z_ref=data._X)
    
    kernel = Matern32(lengthscales=[config['ls']])
    prior = Independent([GP(sparsity=sparsity, kernel=kernel)])
    lik = Gaussian(variance=config['lik_var'])
    
    m = GP(
        data = data,
        prior = prior,
        likelihood = [lik],
        inference = 'Variational'
    )
    
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
        try:
            obj_val, _ = natgrad_trainer.train(1.0, epochs=1)
            grad_step.train(config['lr'], epochs=1)
            learning_curve.append(obj_val[0])
        except Exception as e:
            print(f'Exiting at {i}')
            break
    
    pred_mu, pred_var = m.predict_y(XS)
    
    return {'m': m, 'lc': learning_curve, 'pred_mu': pred_mu, 'pred_var': pred_var}

res_time['vi'] = vi_timeseries_gp(config, XS, X, Y)
