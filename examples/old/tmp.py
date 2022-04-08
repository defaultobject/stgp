import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax

import legogp as lego
from legogp import settings
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer, GradDescentTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32, ScaleKernel
from legogp.likelihood import Gaussian, ReshapedGaussian
from legogp.data import TemporalData, Data, MultiOutputTemporalData
from legogp.models import GP
from legogp.sparsity import NoSparsity, StackedNoSparsity
from legogp.transforms import Independent, DataLatentPermutation
from legogp.transforms.multi_output import LMC
from legogp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, GaussianApproximatePosterior
from legogp.approximate_posteriors import FullGaussianApproximatePosterior

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import numpy as np

from data_zoo import multi_output_timeseries

# Fix randomness
np.random.seed(0)

P = 3
Q = 3

XS, X, Y = multi_output_timeseries(P, 100, 1000, seed=0)

Y_all = np.copy(Y)
#Y[10:30, 2] = np.NaN
config = {
    'ls': [0.1, 0.1, 0.1],
    'lik_var': [0.01, 0.01, 0.01],
    'epochs': 100,
    'lr': 0.01,
    'beta': 0.1,
    'P': P,
    'Q': Q
}

res_time = {}

def mf_vi_lmc(config, XS, X, Y):
    settings.ng_jitter = 1e-8
    data = Data(X, Y)
    
    Z = NoSparsity(Z_ref=data._X)
    
    latents = [
        lego.models.GP(
            sparsity=Z, 
            kernel=Matern32(input_dim=1, lengthscales=[config['ls'][q]], active_dims=[0]),
            latent=True
        ) 
        for q in range(Q)
    ]
    
    prior = LMC(Independent(latents), output_dim = P)
    np.random.seed(0)
    prior._W.raw_var.assign(jax.numpy.array(np.random.randn(3, 3)))
    
    lik = [Gaussian(config['lik_var'][p]) for p in range(P)]

    # get cvi params

    q_cvi = MeanFieldConjugateGaussian([
        ConjugateGaussian(
            X=Z,
            block_size=1,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                data=Data(X=X.raw_Z, Y=Y[..., 0]), 
                prior=Independent([latents[q]]), 
                likelihood=likelihood[0],
                inference='Batch'
            ) # batch gp surrogate model 
        )
        for q in range(Q)
    ])

    cvi_posteriors = [q_cvi.approx_posteriors[q].surrogate.posterior(diagonal=False) for q in range(Q)]

    approx_posterior = MeanFieldApproximatePosterior(approximate_posteriors=[
        GaussianApproximatePosterior(m=cvi_posteriors[q][0][:, None], S=cvi_posteriors[q][1])
        for q in range(Q)
    ])
    
    m = GP(
        data = data,
        prior = prior,
        likelihood=lik,
        inference='Variational',
        approximate_posterior = approx_posterior
    )

    print(m.get_objective())

    if True:
        
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
                obj_val, _ = natgrad_trainer.train(config['beta'], epochs=1)
                grad_step.train(config['lr'], epochs=1)
                learning_curve.append(obj_val[0])
            except Exception as e:
                print(f'Exiting at {i}')
                break
        
        pred_mu, pred_var = m.predict_y(XS)
        
        print(m.get_objective())

        return {'m': m, 'lc': learning_curve, 'pred_mu': pred_mu, 'pred_var': pred_var}

def mf_cvi_batch_lmc(config, XS, X, Y):
    settings.ng_jitter = 1e-8
    data = Data(X, Y)
    
    Z = [NoSparsity(Z_ref=data._X) for q in range(Q)]
    
    latents = [
        lego.models.GP(
            sparsity = Z[q], 
            kernel = Matern32(input_dim=1, lengthscales=[config['ls'][q]], active_dims=[0]),
            latent = True
        ) 
        for q in range(Q)
    ]
    
    prior = LMC(Independent(latents), output_dim = P)
    np.random.seed(0)
    prior._W.raw_var.assign(jax.numpy.array(np.random.randn(3, 3)))
    
    lik = [Gaussian(config['lik_var'][p]) for p in range(P)]
    
    q_cvi = MeanFieldConjugateGaussian([
        ConjugateGaussian(
            X=Z[q],
            block_size=1,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(
                data=Data(X=X.raw_Z, Y=Y[..., 0]), 
                prior=Independent([latents[q]]), 
                likelihood=likelihood[0],
                inference='Batch'
            ) # batch gp surrogate model 
        )
        for q in range(Q)
    ])

    m = GP(
        data = data,
        prior = prior,
        likelihood=lik,
        approximate_posterior=q_cvi,
        inference='Variational'
    )
    print(m.get_objective())

    if True:
        
        # Natgrad step
        natgrad_trainer = NatGradTrainer(m, schedule='constant')
        grad_step = GradDescentTrainer(m, objax.optimizer.Adam)
        
        learning_curve = []
        for i in range(config['epochs']):
            try:
                obj_val, _ = natgrad_trainer.train(config['beta'], epochs=1)
                grad_step.train(config['lr'], epochs=1)
                learning_curve.append(obj_val[0])
            except Exception as e:
                print(e)
                print(f'Exiting at {i}')
                break
        
        pred_mu, pred_var = m.predict_y(XS)

        print(m.get_objective())
        
        return {'m': m, 'lc': learning_curve, 'pred_mu': pred_mu, 'pred_var': pred_var}
        
res_time['mf_batch_cvi'] = mf_cvi_batch_lmc(config, XS, X, Y)
res_time['mf_vi'] = mf_vi_lmc(config, XS, X, Y)
