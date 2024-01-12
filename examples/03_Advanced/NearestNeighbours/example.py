import sys
sys.path.append('../')
sys.path.append('../../')

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp.data import Data

from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.standard import ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, Matern52, ScaledMatern52
from stgp.likelihood import Gaussian, ProductLikelihood, GaussianProductLikelihood
from stgp.transforms import Independent, PrecomputedNearestNeighbours, DataStack
from stgp.approximate_posteriors import MeanFieldConjugateGaussian, ConjugateGaussian, ConjugatePrecisionGaussian, GaussianApproximatePosterior, MeanFieldApproximatePosterior, MeanFieldAcrossDataApproximatePosterior
from stgp.zoo.gps import batch_gp
from stgp import settings
from stgp.models import GP
from stgp.sparsity import StackedSparsity, FullSparsity, NoSparsity
from stgp.data import PrecomputedGroupedNearestNeighboursData

import numpy as np

from example_utils.data_zoo import single_output_timeseries

import matplotlib.pyplot as plt

XS, X, Y = single_output_timeseries(100, 1000, seed=0)

#create 10 groups of 10
num_groups = 10
group_size = 10
group_indexes = np.arange(X.shape[0]).reshape([num_groups, group_size])
data_group_map = np.reshape(np.tile(np.arange(num_groups)[:, None], [1, group_size]), [-1, 1])

Q = 1
sparsity =[
    FullSparsity(Z = X[group_indexes[i]])
    for i in range(num_groups)
]

# same kernel across all independent groups
kern = ScaleKernel(Matern52(input_dim=1, lengthscales=[0.1]))
latent_gps = [
    GP(sparsity=sparsity[i], kernel=kern, prior=True)
    for i in range(num_groups)
]

if False:
    Z = StackedSparsity(sparsity).Z

    k = 5
    distance_matrix = jax.vmap(jax.vmap(lambda a, b: jax.numpy.sum(jax.numpy.square(a-b)), [None, 0]), [0, None])(X, Z)
    distance_matrix_sorted = jax.numpy.argsort(distance_matrix, axis=-1)
    knn_indices = distance_matrix_sorted[..., :k]

    if False:
        data = PrecomputedGroupedNearestNeighboursData(
            Data(X, Y),
            knn_indices
        )
    else:
        data = Data(X, Y)

    m = GP(
        data = data,
        prior = PrecomputedNearestNeighbours(
            Independent(latent_gps),
            data_group_map = data_group_map,
            groups = group_indexes
        ),
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior = MeanFieldApproximatePosterior(
            approximate_posteriors = [
                GaussianApproximatePosterior(dim=group_size)
                for i in range(num_groups)
            ] 
        ),
        inference='Variational'
    )
elif False:

    k = 5
    distance_matrix = jax.vmap(jax.vmap(lambda a, b: jax.numpy.sum(jax.numpy.square(a-b)), [None, 0]), [0, None])(X, X)
    distance_matrix_sorted = jax.numpy.argsort(distance_matrix, axis=-1)
    knn_indices = distance_matrix_sorted[..., :k]

    kern = ScaleKernel(Matern52(input_dim=1, lengthscales=[0.1]))
    latent_gps = [ GP(sparsity=NoSparsity(X), kernel=kern, prior=True) ]

    m = GP(
        data = Data(X, Y),
        prior = PrecomputedNearestNeighbours(Independent(latent_gps), knn_indices, None),
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1)]),
        inference='Batch'
    )

else:
    Q = 1
    group_size = X.shape[0]
    num_groups = int(X.shape[0]/group_size)
    print('group_size: ', group_size)
    print('num_groups: ', num_groups)
    sparsity =[
        FullSparsity(Z = X[i*group_size:(i*group_size+group_size), :])
        for i in range(num_groups)
    ]
    print(sparsity[-1].Z)

    # same kernel across all independent groups
    kern = ScaleKernel(Matern52(input_dim=1, lengthscales=[0.1]))
    latent_gps = [
        GP(sparsity=sparsity[i], kernel=kern, prior=True)
        for i in range(num_groups)
    ]


    m = GP(
        data = Data(X, Y),
        prior = Independent([DataStack(latent_gps)]),
        likelihood = GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior = MeanFieldApproximatePosterior(
            approximate_posteriors = [
                MeanFieldAcrossDataApproximatePosterior(
                    approximate_posteriors = [
                        GaussianApproximatePosterior(dim=group_size)
                        for i in range(num_groups)
                    ] 
                )
            ]
        ),
        inference='Variational'
    )

if True:
    if True:
        lc_arr, _ = NatGradTrainer(m).train(0.1, 1)

        lc_arr, _ = ADAM(m).train(0.01, 1000, callback=progress_bar_callback(1000))
        m.checkpoint('check')
        plt.plot(lc_arr)
        plt.yscale('log')
        plt.show()
    else:
        m.load_from_checkpoint('check')

print(m.get_objective())
pred_mu, pred_var = m.predict_f(XS)

plt.fill_between( np.squeeze(XS), np.squeeze(pred_mu) - 2*np.sqrt(np.squeeze(pred_var)), np.squeeze(pred_mu) + 2*np.sqrt(np.squeeze(pred_var)), alpha=0.4)
plt.plot(np.squeeze(XS), np.squeeze(pred_mu))
plt.scatter(np.squeeze(X), np.squeeze(Y))
plt.show()
