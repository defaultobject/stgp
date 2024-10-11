""" Conjugate Variational Gaussian Process Regression on a temporal dataset"""
import sys
sys.path.append('../')
sys.path.append('../../')

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import multi_output_spatial_data
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, Matern52, ScaledMatern52, SpatioTemporalSeperableKernel
from stgp.likelihood import Gaussian, ProductLikelihood, GaussianProductLikelihood
from stgp.data import Data, SpatioTemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import MeanFieldConjugateGaussian, ConjugateGaussian, ConjugatePrecisionGaussian, FullGaussianApproximatePosterior, FullConjugatePrecisionGaussian, FullConjugateGaussian
from stgp.sparsity import StackedSparsity, StackedNoSparsity
from stgp import settings

from tqdm import trange

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

settings.jitter = 1e-8
settings.ng_jitter = 1e-8

# Construct Data
P = 3
Q = 3

NS = 20
XS, X, Y = multi_output_spatial_data(P, 10, 10, NS, NS, seed=0)


print(f'X: {X.shape}, Y: {Y.shape}')

def batch_lmc():
    # Construct Model
    Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

    # Construct Latent GPs
    latent_kernels = [
        ScaleKernel(Matern52(input_dim=2, lengthscales=[0.1, 0.1])) for q in range(Q)
    ]

    latent_gps = [
        stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
    ] 

    np.random.seed(0)
    W =  np.random.randn(P, Q)
    print('W: ', W)

    prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim = P, input_dim=Q, W = W)

    m = stgp.models.GP(
        data=Data(X, Y), 
        likelihood=[Gaussian(0.1) for p in range(P)],
        inference='Batch',
        prior=prior,
    )

    print(m.get_objective())
    m.print()
    return m


def vi_lmc(mf=False):
    # Construct Model
    Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

    # Construct Latent GPs
    latent_kernels = [
        ScaleKernel(Matern52(input_dim=2, lengthscales=[0.1, 0.1])) for q in range(Q)
    ]

    latent_gps = [
        stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
    ] 

    np.random.seed(0)
    W =  np.random.randn(P, Q)
    #W =  np.eye(P)
    print('W: ', W)

    prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim = P, input_dim=Q, W = W)

    if mf:
        m = stgp.models.GP(
            data=Data(X, Y), 
            likelihood=[Gaussian(0.1) for p in range(P)],
            inference='Variational',
            prior=prior
        )
    else:
        m = stgp.models.GP(
            data=Data(X, Y), 
            likelihood=[Gaussian(0.1) for p in range(P)],
            inference='Variational',
            prior=prior,
            approximate_posterior = FullGaussianApproximatePosterior(dim = X.shape[0] * prior.base_prior.output_dim)
        )

    print(m.get_objective())
    m.print()
    from stgp.computation.model_ops import get_diagonal_gaussian_likelihood_variances
    from stgp.computation.natural_gradients import exponential_family_transforms
    from stgp.computation.matrix_ops import vec_columns, get_block_diagonal, to_block_diag
    # latent - data form
    Kxx = m.prior.base_prior.covar(X, X)
    mean = m.prior.base_prior.mean(X)

    # latent - data form
    lik_var = get_diagonal_gaussian_likelihood_variances(Y, m.likelihood.likelihood_arr)

    # latent - data
    W1 = jax.numpy.kron(m.prior.W, np.eye(X.shape[0]))

    # latent - data
    Y_vec = vec_columns(Y)

    # nat params of prior
    prior_nat_params = exponential_family_transforms.theta_to_lambda(mean, Kxx)
    likelihood_nat_params_no_W = exponential_family_transforms.theta_to_lambda(Y_vec, lik_var)
    # TODO: fix transposes in the proof -.-
    likelihood_nat_params = [W1.T @ likelihood_nat_params_no_W[0], W1.T @ likelihood_nat_params_no_W[1] @ W1]

    # nat params of full posterrior 
    posterior_nat_params = [likelihood_nat_params[0] + prior_nat_params[0], likelihood_nat_params[1] + prior_nat_params[1]]
    # approximate posterior nat params with block diagonal

    posterior_nat_params = [posterior_nat_params[0], to_block_diag(get_block_diagonal(posterior_nat_params[1], X.shape[0]))]

    posterior_moment_params = exponential_family_transforms.lambda_to_theta(posterior_nat_params[0], posterior_nat_params[1])

    ng_trainer = NatGradTrainer(m)
    ng_trainer.train(1.0, 1)

    q_m = m.approximate_posterior.m
    q_m = q_m.reshape([-1, 3])

    test_m = posterior_moment_params[0]
    test_m = test_m.reshape([-1, 3])
    for i in range(3):
        plt.plot(q_m[:, i], color='green')
        plt.plot(test_m[:, i], color='red')
    plt.show()
    breakpoint()
    return m
    

def cvi_gp_lmc(parameterisation):
    """ 
    CVI-LMC with a standard GP surrogate model
    Args:
        parameterisation:
            [covariance] - the surrogate GP likelihood will be parameterised using covariances
            [precision] - the surrogate GP likelihood will be parameterised using precisions
    """
    # Construct Model
    Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]
    sparsity = StackedNoSparsity(Z)

    # Construct Latent GPs
    latent_kernels = [
        ScaleKernel(Matern52(input_dim=2, lengthscales=[0.1, 0.1])) for q in range(Q)
    ]

    latent_gps = [
        stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
    ] 

    np.random.seed(0)
    W =  np.random.randn(P, Q)
    print('W: ', W)

    prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim = P, input_dim=Q, W = W)


    if parameterisation == 'covariance':
        cvi_class = FullConjugateGaussian
    elif parameterisation == 'precision':
        cvi_class = FullConjugatePrecisionGaussian
    else:
        raise NotImplementedError()

    data = Data(X, Y)
    m = GP(
        data = data,
        prior = prior,
        likelihood=[Gaussian(0.1) for p in range(P)],
        approximate_posterior = cvi_class(
            X=data.X,
            num_blocks = data.N,
            block_size=Q,
            num_latents=Q,
            surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                data=Data(X, Y), # Data should already be in the correct format
                prior=Independent(latent_gps), 
                likelihood=likelihood
            )  
        ),
        inference='Variational'
    )
    m.print()
    return m

def cvi_sde_gp(parallel=False, parameterisation='covariance'):
    """ CVI-GP with a state-space GP surrogate model parameterised using moment parameterisation.  """
    settings.cvi_ng_exploit_space_time = True

    # Construct Model
    Q = 3
    st_data = SpatioTemporalData(X=X, Y=Y, sort=True)
    sparsity = stgp.sparsity.NoSparsity(Z = st_data.X)

    kern = SpatioTemporalSeperableKernel(
        ScaledMatern52(input_dim=1, lengthscales=[0.1], variance=1.0, active_dims=[0]),
        Matern52(input_dim=1, lengthscales=[0.1],  active_dims=[1])
    )
    latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True) for q in range(Q)]

    if parameterisation == 'covariance':
        cvi_class = ConjugateGaussian
    elif parameterisation == 'precision':
        cvi_class = ConjugatePrecisionGaussian
    else:
        raise NotImplementedError()

    data = Data(X, Y)

    np.random.seed(0)
    W =  np.random.randn(P, Q)
    print('W: ', W)

    prior = stgp.transforms.multi_output.LMC(Independent(latent_gps), output_dim = P, input_dim=Q, W = W)

    m = GP(
        data = st_data,
        prior = prior,
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1) for p in range(P)]),
        approximate_posterior = MeanFieldConjugateGaussian([
            cvi_class(
                X=st_data._X,
                num_blocks = st_data.Nt,
                block_size=st_data.Ns,
                num_latents=1,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    data=SpatioTemporalData(X=X, Y=np.reshape(Y, [st_data.Nt, 1, st_data.Ns]), sort=False), # Data should already be in the correct format
                    prior=LTI_SDE(Independent([latent_gps[q]])), 
                    likelihood=likelihood,
                    inference='Sequential',
                    filter_typr='sequential'
                )  
            )
            for q in range(Q)
        ]),
        inference='Variational'
    )
    return m

vi_lmc(mf=True)
breakpoint()

models = {
    'batch': batch_lmc(),
    'vi_fp': vi_lmc(mf=False),
    'vi_mf': vi_lmc(mf=True),
    #'cvi_gp_lmc_covariance': cvi_gp_lmc(parameterisation='covariance'),
    'cvi_gp_lmc_covariance': cvi_sde_gp(),
    #'cvi_gp_lmc_precision': cvi_gp_lmc(parameterisation='precision'),
}

if True:
    for k, m in models.items():
        if k == 'batch': 
            # no need for batch models
            continue

        try:
            ng_trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton_delta_u')
            ng_trainer.train(1.0, 1)
        except RuntimeError as e:
            ng_trainer = NatGradTrainer(m)
            ng_trainer.train(1.0, 1)
        print(m.get_objective())

# check that the ELBOs remain the same after
if True:
    for k, m in models.items():
        print(f'{k}: {m.get_objective()}')


N_models = len(models)

fig, axes = plt.subplots(N_models, P, squeeze=False)

print('predicting')

for i, key   in enumerate(models):
    m = models[key]

    ax_i = axes[i]

    pred_mu, pred_var = m.predict_f(XS)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)

    print(np.sum(pred_mu), np.sum(pred_var))

    for p in range(P):
        ax_i[p].set_title(f'Output {p}')
        ax_i[p].imshow(pred_mu[:, p].reshape(NS, NS))

plt.show()


