import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax

import stgp as lego
from stgp import settings
from stgp.trainers import ScipyTrainer, NatGradTrainer, GradDescentTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32, ScaleKernel
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.data import TemporalData, Data, MultiOutputTemporalData
from stgp.models import GP
from stgp.sparsity import NoSparsity, StackedNoSparsity
from stgp.transforms import Independent, DataLatentPermutation
from stgp.transforms.multi_output import LMC, GPRN_DRD
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
from stgp.metrics.nlpd import nlpd
from stgp.transforms.sdes import LTI_SDE


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
    
    prior = LMC(Independent(latents), output_dim = P)
    
    lik = [Gaussian(config['lik_var'][p]) for p in range(P)]
    
    q_cvi = FullConjugateGaussian(
        X=Z_all,
        num_latents=Q,
        block_size=Q,
        surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
            data=MultiOutputTemporalData(X=X.sparsity_arr[0], Y=Y[:, None, :], sort=False), # Data should already be in the correct format
            prior=LTI_SDE(prior.latent_obj), 
            likelihood=likelihood,
            inference='Sequential'
        )  
    )

    m = GP(
        data = data,
        prior = prior,
        likelihood=lik,
        approximate_posterior=q_cvi,
        inference='Variational',
        prediction_samples=1000
    )
    
    # Natgrad step
    natgrad_trainer = NatGradTrainer(m, schedule='constant')
    
    grad_step = GradDescentTrainer(m, objax.optimizer.Adam)
    
    learning_curve = []
    for i in range(config['epochs']):
        obj_val, _ = natgrad_trainer.train(1.0, epochs=1)
        learning_curve.append(obj_val[0])
        grad_step.train(config['lr'], epochs=1)
    
    pred_mu, pred_var = m.predict_y(XS)
    
    return {'m': m, 'lc': learning_curve, 'pred_mu': pred_mu, 'pred_var': pred_var}    

res_time['dense_sde_cvi'] = dense_sde_cvi_lmc(config, XS, X, Y)

