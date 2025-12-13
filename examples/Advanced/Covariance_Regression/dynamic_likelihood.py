import sys

sys.path.append("../../")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
import numpy as np

from example_utils.data_zoo import multi_output_timeseries
from stgp.trainers import ScipyTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF
from stgp.likelihood.dynamic_covariance_likelihood import DynamicCovarianceGaussian
from stgp.data import DataList

from stgp.transforms.covariance import LKJStaticVarianceProcess
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
import stgp
from stgp.transforms import OutputMap

import matplotlib.pyplot as plt

stgp.settings.use_loop_mode = True

# Generate data
Q = 3
P = 3
N = 10

XS, X, Y = multi_output_timeseries(P, N, 500, seed=0)

X_test = np.copy(X)
Y_test = np.copy(Y)


# Construct Model
Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

# Construct Latent GPs
num_W = int(P * (Q - 1) / 2)

latent_W_gps = [
    stgp.models.GP(sparsity=stgp.sparsity.NoSparsity(X), kernel=RBF(lengthscales=[0.1]))
    for q in range(num_W)
]

latent_prior = LKJStaticVarianceProcess(latent_W_gps, input_dim=P, output_dim=P)

prior = OutputMap(
    latent_prior,
    [[[0, 1, 2]]],  # concat
)

# one likelihood
m = stgp.models.GP(
    data=DataList(X, [Y]),
    likelihood=[DynamicCovarianceGaussian()],
    inference="Variational",
    prior=prior,
    ell_samples=10,
    approximate_posterior=FullGaussianApproximatePosterior(
        dim=X.shape[0] * prior.base_prior.output_dim
    ),
)
print(m.get_objective())

if False:
    epochs = 1000
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = ScipyTrainer(m, "L-BFGS-B").train(
        None, epochs, callback=callback
    )

    # Plot learning curve
    plt.plot(learning_curve)
    plt.yscale("log")
    plt.show()

print(m.samples(XS, num_samples=10))
print(m.nlpd(X, Y, num_samples=100))
breakpoint()
print(m.predict_f(XS, num_samples=100))
breakpoint()
