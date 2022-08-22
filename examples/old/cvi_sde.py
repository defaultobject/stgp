import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax

import stgp as lego
from stgp import settings
from stgp.trainers import ScipyTrainer, NatGradTrainer, GradDescentTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.data import TemporalData, Data
from stgp.models import GP
from stgp.sparsity import NoSparsity
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian


import matplotlib.pyplot as plt

import numpy as np

from data_zoo import single_output_timeseries

# Fix randomness
np.random.seed(0)

XS, X, Y = single_output_timeseries(100, 1000, seed=0)

config = {
    'ls': 0.1,
    'lik_var': 0.001,
    'epochs': 1,
    'lr': 0.01
}

res_time = {}

def cvi_t_sde_gp(config, XS, X, Y):
    settings.ng_jitter = 1e-5
    settings.jitter = 1e-5
    
    Q = 1
    
    data = Data(X=X, Y=Y)
    sparsity = [NoSparsity(Z_ref=data._X)]
    
    kernel = Matern32(lengthscales=[config['ls']])
    latent_gps = [GP(sparsity=sparsity[0], kernel=kernel)]
    lik = Gaussian(variance=config['lik_var'])
    
    approx_posterior = MeanFieldConjugateGaussian([
        ConjugateGaussian(
            X=sparsity[q],
            block_size=1,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                data=TemporalData(X=X.raw_Z, Y=Y, sort=False), # Data should already be in the correct format
                prior=LTI_SDE(Independent([latent_gps[q]])), 
                likelihood=likelihood[q],
                inference='Sequential'
            )  
        )
        for q in range(Q)
    ])
    
    m = GP(
        data = data,
        prior = Independent(latent_gps),
        likelihood = [lik],
        approximate_posterior=approx_posterior,
        inference = 'Variational'
    )
    
    # Natgrad step
    natgrad_trainer = NatGradTrainer(m, schedule='constant')
    # when using a step size of 1.0 we dont need to remove params from adam
    grad_step = GradDescentTrainer(m, objax.optimizer.Adam)

    learning_curve = []
    for i in range(config['epochs']):
        obj_val, _ = natgrad_trainer.train(1.0, epochs=1)
        grad_step.train(config['lr'], epochs=1)
        learning_curve.append(obj_val[0])

    print(objax.Grad(m.get_objective, m.vars())())
    print(objax.Grad(m.approximate_posterior.approx_posteriors[0].surrogate.get_objective, m.vars())())
    print(m.approximate_posterior.approx_posteriors[0].surrogate.predict_f(data.X)[0].shape)

    breakpoint()
    
    pred_mu, pred_var = m.predict_y(XS)
    
    return {'m': m, 'lc': learning_curve, 'pred_mu': pred_mu, 'pred_var': pred_var}

res_time['cvi_t_sde'] = cvi_t_sde_gp(config, XS, X, Y)

