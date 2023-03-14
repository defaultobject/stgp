""" Conjugate Variational Gaussian Process Regression on a temporal dataset"""
import sys
sys.path.append('../')
sys.path.append('../../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, Matern52, ScaledMatern52, SpatioTemporalSeperableKernel
from stgp.likelihood import Gaussian, ProductLikelihood, GaussianProductLikelihood
from stgp.data import Data, SpatioTemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import MeanFieldConjugateGaussian, ConjugateGaussian, ConjugatePrecisionGaussian

from tqdm import trange

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

# Construct Data
NS = 100
XS, X, Y = single_output_spatial_data(10, 10, NS, NS, seed=0)

print(f'X: {X.shape}, Y: {Y.shape}')

def cvi_gp(parameterisation):
    """ 
    CVI-GP with a standard GP surrogate model
    Args:
        parameterisation:
            [covariance] - the surrogate GP likelihood will be parameterised using covariances
            [precision] - the surrogate GP likelihood will be parameterised using precisions
    """
    # Construct Model
    Q = 1
    sparsity = stgp.sparsity.NoSparsity(Z = X)

    kern = ScaleKernel(Matern52(input_dim=2, lengthscales=[0.1, 0.1]))
    latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True)]

    if parameterisation == 'covariance':
        cvi_class = ConjugateGaussian
    elif parameterisation == 'precision':
        cvi_class = ConjugatePrecisionGaussian
    else:
        raise NotImplementedError()

    data = Data(X, Y)
    m = GP(
        data = data,
        prior = Independent(latent_gps),
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior = MeanFieldConjugateGaussian([
            cvi_class(
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
    print(m.get_objective())
    return m

def cvi_sde_gp(parallel=False, parameterisation='covariance'):
    """ CVI-GP with a state-space GP surrogate model parameterised using moment parameterisation.  """
    # Construct Model
    Q = 1
    st_data = SpatioTemporalData(X=X, Y=Y, sort=True)
    sparsity = stgp.sparsity.NoSparsity(Z = st_data.X)

    kern = SpatioTemporalSeperableKernel(
        ScaledMatern52(input_dim=1, lengthscales=[0.1], variance=1.0, active_dims=[0]),
        Matern52(input_dim=1, lengthscales=[0.1],  active_dims=[1])
    )
    latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True)]

    if parameterisation == 'covariance':
        cvi_class = ConjugateGaussian
    elif parameterisation == 'precision':
        cvi_class = ConjugatePrecisionGaussian
    else:
        raise NotImplementedError()

    data = Data(X, Y)

    m = GP(
        data = st_data,
        prior = Independent(latent_gps),
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior = MeanFieldConjugateGaussian([
            cvi_class(
                X=st_data._X,
                num_blocks = st_data.Nt,
                block_size=st_data.Ns,
                num_latents=1,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    data=SpatioTemporalData(X=X, Y=np.reshape(Y, [st_data.Nt, 1, st_data.Ns]), sort=False), # Data should already be in the correct format
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
    print(m.get_objective())
    return m


models = {
    'cvi_gp_precision': cvi_gp(parameterisation='precision'),
    'cvi_gp': cvi_gp(parameterisation='covariance'),
    'cvi_sde_gp_seq_precision': cvi_sde_gp(parallel=False, parameterisation='precision'),
    'cvi_sde_gp_seq_covariance': cvi_sde_gp(parallel=False, parameterisation='covariance'),
    'cvi_sde_gp_parallel_precision': cvi_sde_gp(parallel=True, parameterisation='precision'),
    'cvi_sde_gp_parallel_covariance': cvi_sde_gp(parallel=True, parameterisation='covariance'),
}

if True:
    for k, m in models.items():
        ng_trainer = NatGradTrainer(m)
        ng_trainer.train(1.0, 1)

# check that the ELBOs remain the same after
if True:
    for k, m in models.items():
        print(f'{k}: {m.get_objective()}')

N_models = len(models)

fig, axes = plt.subplots(N_models, 2, squeeze=False)

for i, key   in enumerate(models):
    m = models[key]

    ax_i = axes[i]

    pred_mu, pred_var = m.predict_f(XS)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)

    pred_mu = pred_mu.reshape(NS, NS)
    pred_var = pred_var.reshape(NS, NS)

    ax_i[0].set_title('Mean')
    ax_i[0].imshow(pred_mu)


    ax_i[1].set_title('Var')
    ax_i[1].imshow(pred_var)
plt.show()

