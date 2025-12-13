import sys

sys.path.append("../../")

from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_disable_jit", False)
import objax
import numpy as np

from example_utils.data_zoo import multi_output_timeseries
from example_utils import colors
from stgp.trainers import GradDescentTrainer, NatGradTrainer
from stgp.kernels import RBF
from stgp.likelihood import Gaussian
from stgp.data import Data
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
import stgp

from tqdm import trange

import matplotlib.pyplot as plt

# Generate data
Q = 3
P = 3
N = 50

XS, X, Y = multi_output_timeseries(P, N, 10, seed=0)

X_test = np.copy(X)
Y_test = np.copy(Y)

# Remove section of Y for testing
Y[20:35, 1] = np.NaN

print(f"X: {X.shape}, Y: {Y.shape}")

if False:
    for p in range(P):
        plt.plot(X[:, 0], Y[:, p])
    plt.show()

# Construct Model
Z = np.linspace(np.min(X), np.max(X), 35)[:, None]

# use same Z for now
Z_list = [stgp.sparsity.FullSparsity(Z) for q in range(Q)]

# Construct Latent GPs
latent_kernels = [RBF(lengthscales=[0.1]) for q in range(Q)]
latent_gps = [
    stgp.models.GP(sparsity=Z_list[q], kernel=latent_kernels[q]) for q in range(Q)
]
np.random.seed(0)
W = np.random.randn(P, Q)
print("W: ", W)

prior = stgp.transforms.multi_output.LMC(latent_gps, output_dim=P, input_dim=2, W=W)

m = stgp.models.GP(
    data=Data(X, Y),
    likelihood=[Gaussian(0.1), Gaussian(0.1), Gaussian(0.1)],
    inference="Variational",
    prior=prior,
    approximate_posterior=FullGaussianApproximatePosterior(
        dim=Z.shape[0] * prior.base_prior.output_dim
    ),
)


print(m.get_objective())
print("NLPD: ", m.nlpd(X, Y, num_samples=10000))


pred_mu, pred_var = m.predict_f(XS)

if False:
    max_iters = 200

    ng_trainer = NatGradTrainer(m)
    # ng_trainer = NatGradTrainer(m, enforce_psd_type='gauss_newton', prediction_samples=100)
    # ng_trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton')
    m.approximate_posterior.fix()

    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    ng_trainer.train(1.0, 1)
    if True:
        print(m.get_objective())

        for i in trange(max_iters):
            trainer.train(0.01, 1)
            ng_trainer.train(1.0, 1)

        print(m.get_objective())
else:
    ng_trainer = NatGradTrainer(m)
    ng_trainer.train(1.0, 1)

    # lc, _ = GradDescentTrainer(m, objax.optimizer.Adam).train(0.01, 100)
    # plt.plot(lc)
    # plt.show()
    # m.print()

print("NLPD: ", m.nlpd(X, Y, num_samples=10000))

pred_mu, pred_var = m.predict_y(XS, diagonal=True, output_first=True)

fig, axes = plt.subplots(P, 1, sharex=True)

Z_list = m.prior.get_sparsity_list()
m_list = m.approximate_posterior.m
M = Z_list[0].Z.shape[0]

for p in range(P):
    m_p = m_list[p * M : p * M + M]
    Z_p = np.squeeze(Z_list[p].Z)

    axes[p].fill_between(
        np.squeeze(XS),
        np.squeeze(pred_mu[p] - 1.96 * np.sqrt(pred_var[p])),
        np.squeeze(pred_mu[p] + 1.96 * np.sqrt(pred_var[p])),
        facecolor=colors.LINE_COL,
        alpha=0.3,
    )
    axes[p].plot(XS, pred_mu[p], color=colors.LINE_COL, label="GP Fit")
    axes[p].scatter(X_test, Y_test[:, p], color="grey", label="Testing Data")
    axes[p].scatter(X, Y[:, p], color="lightgrey", label="Training Data")

    # this plots the inducing points of the latent processes
    # axes[p].scatter(Z_p, np.squeeze(m_p), 1, label='Inducing Points', color='black')
    # axes[p].scatter(Z_p, np.zeros_like(np.squeeze(m_p)), 0.5)

    axes[p].legend()
plt.show()
