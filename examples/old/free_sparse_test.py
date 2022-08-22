import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)
import objax

import stgp as lego
from stgp import settings
from stgp.trainers import ScipyTrainer, NatGradTrainer, GradDescentTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32, ScaleKernel, RBF
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.data import TemporalData, Data, MultiOutputTemporalData
from stgp.models import GP
from stgp.sparsity import NoSparsity, StackedNoSparsity, FullSparsity
from stgp.transforms import Independent, DataLatentPermutation
from stgp.transforms.multi_output import LMC
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from stgp.approximate_posteriors import FullGaussianApproximatePosterior

import matplotlib.pyplot as plt

import numpy as np

from data_zoo import single_output_timeseries

np.random.seed(0)


XS, X, Y = single_output_timeseries(100, 1000, seed=0)

config = {
    'ls': [0.1, 0.1, 0.1],
    'lik_var': [0.01, 0.1, 0.1],
    'epochs': 10,
    'lr': 0.01,
    'beta': 1.0
}

res_time = {}

def svgp(config, XS, X, Y):
    settings.jitter = 1e-7
    data = Data(X, Y, minibatch_size=None)

    M = 10
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

    #print(m.get_objective())
    
    natgrad_trainer = NatGradTrainer(m, schedule='constant')

    # Do not use adam for the approx posterios
    all_vars = list(m.vars().keys())
    m_name = [a for a in all_vars if a.endswith('._m(Parameter).raw_var')]
    s_chol_name = [a for a in all_vars if a.endswith('._S_chol(Parameter).raw_var')]
    approx_posterior_vars = m_name + s_chol_name
    grad_step = GradDescentTrainer(m, objax.optimizer.Adam, hold_vars = approx_posterior_vars)

    learning_curve = []
    for i in range(config['epochs']):
        obj_val, _ = natgrad_trainer.train(config['beta'], epochs=1)
        #grad_step.train(config['lr'], epochs=1)
        learning_curve.append(obj_val[0])

    print(m.get_objective())
    breakpoint()
    
    return {'m': m, 'lc': learning_curve}
    



def gpflow_model(config, XS, X, Y):
    import gpflow
    from gpflow.models import VGP, GPR, SGPR, SVGP
    from gpflow.optimizers import NaturalGradient
    from gpflow.optimizers.natgrad import XiSqrtMeanVar


    M = 10
    Z = np.linspace(0, 1, M)[:, None]


    inducing_variable = Z

    data = (X, Y)

    svgp = SVGP(
        kernel=gpflow.kernels.RBF(lengthscales=config['ls'][0]),
        likelihood=gpflow.likelihoods.Gaussian(variance=config['lik_var'][0]),
        inducing_variable=inducing_variable,
        whiten=False
    )

    print(-svgp.elbo(data).numpy())

    if True:
        natgrad_opt = NaturalGradient(gamma=1e-3)
        variational_params = [(svgp.q_mu, svgp.q_sqrt, XiSqrtMeanVar())]
        svgp_natgrad_loss = svgp.training_loss_closure(data)
        for i in range(100):
            natgrad_opt.minimize(svgp_natgrad_loss, var_list=variational_params)


        print(-svgp.elbo(data).numpy())
        natgrad_opt = NaturalGradient(gamma=0.1)
        variational_params = [(svgp.q_mu, svgp.q_sqrt, XiSqrtMeanVar())]
        svgp_natgrad_loss = svgp.training_loss_closure(data)
        natgrad_opt.minimize(svgp_natgrad_loss, var_list=variational_params)
    else:
        natgrad_opt = NaturalGradient(gamma=1.0)
        variational_params = [(svgp.q_mu, svgp.q_sqrt)]
        svgp_natgrad_loss = svgp.training_loss_closure(data)
        natgrad_opt.minimize(svgp_natgrad_loss, var_list=variational_params)

    print(-svgp.elbo(data).numpy())

    return {'m':svgp, 'lc': None}
if True:
    res_time['batch'] = svgp(config, XS, X, Y)
    m = res_time['batch']['m']
    lc = res_time['batch']['lc']
else:
    res_time['gpflow'] = gpflow_model(config, XS, X, Y)
    m = res_time['gpflow']['m']
    lc = res_time['gpflow']['lc']


plt.plot(lc)
plt.show()

pred_mu, pred_var = m.predict_f(XS)

plt.fill_between(np.squeeze(XS), np.squeeze(pred_mu + 1.96 * np.sqrt(pred_var)), np.squeeze(pred_mu - 1.96 * np.sqrt(pred_var)), alpha = 0.4)
plt.plot(XS, pred_mu)
plt.scatter(X, Y)
plt.show()
