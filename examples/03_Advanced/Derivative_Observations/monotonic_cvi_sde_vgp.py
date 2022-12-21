""" 
Variational Monotonic Gaussian Process with Virtual Monotonic Observations and a SDE CVI Surrogate Posterior

Monotonicitiy enforced following:
    Gaussian processes with monotonicity information, Riihim ̈aki et all
    http://proceedings.mlr.press/v9/riihimaki10a/riihimaki10a.pdf
"""


import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp

import objax

import numpy as np

import stgp
from stgp import settings
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, ScaledMatern32, Matern52, ScaledMatern52
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.likelihood import Gaussian, Probit, ProductLikelihood
from stgp.models import GP
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.data import Data
from stgp.approximate_posteriors import FullConjugateGaussian
from tqdm import trange
from stgp.trainers import NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs
from stgp.transforms import Independent

import matplotlib.pyplot as plt


settings.ng_jitter = 1e-4
f = lambda x: 20*np.sin(100*x)+100*x

N = 20
x = np.linspace(0, 1, N)
y = f(x) + 0.5*np.random.rand(N)

X = x[:, None]
Y = y[:, None]
F = f(x)[:, None]

XS = np.linspace(0, 1, 1000)[:, None]

# add virtual observations (not actually used in the likelihood but have to be passed)

Y = np.hstack([Y, np.ones_like(Y)])

N_colocation = 200
X_colocation = np.linspace(np.min(X), np.max(X), N_colocation)[:, None]
Y_colocation = np.ones([N_colocation, Y.shape[1]])
Y_colocation[:, 0] = np.NaN

X = np.vstack([X, X_colocation])
Y = np.vstack([Y, Y_colocation])

# sort data
st_data = stgp.data.MultiOutputTemporalData(X, Y)
X, Y = st_data.X, st_data.Y

if False:
    #plt.plot(X, F)
    plt.scatter(X, Y[:, 0])
    plt.scatter(X, Y[:, 1])
    plt.show()

base_kernel_1d = ScaledMatern32(input_dim = 1, lengthscales = [1.0], variance=1.0)
kern = FirstOrderDerivativeKernel(base_kernel_1d)
sparsity = stgp.sparsity.NoSparsity(Z=X)

base_gp =  GP(
    sparsity=sparsity, 
    kernel = base_kernel_1d
)

prior = DifferentialOperatorJoint(
    base_gp,
    kernel = kern,
    is_base = True,
    has_parent=False
)

# use full gaussian for consistency
q = FullConjugateGaussian(
    X = sparsity,
    num_latents=2,
    block_size=2,
    num_blocks = X.shape[0],
    surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
        data = stgp.data.MultiOutputTemporalData(X=X, Y=Y[:, :, None], sort=False, train_y=True), # we need gradients Y so set to be trainable
        likelihood=likelihood, 
        prior=LTI_SDE_Full_State_Obs(Independent([base_gp])),
        inference='Sequential',
        full_state_observed = True
    )
)

gauss_lik = Gaussian(0.01)
gauss_lik.fix()

# Create Model
m = stgp.models.GP(
    data = stgp.data.Data(X, Y),
    prior = prior,
    likelihood = [gauss_lik, Probit(nu=1.0)],
    approximate_posterior = q,
    ell_samples = 100,
    prediction_samples = 1000,
    whiten=False,
    inference='Variational'
)

print(m.get_objective())


# Train
if True:
    print(m.get_objective())
    max_iters = 2000
    ng_trainer = NatGradTrainer(m, enforce_psd_type='retraction')
    #ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    lc_arr_1, _ = ng_trainer.train(0.001, 100)
    lc_arr = np.array(lc_arr_1).tolist()

    if True:
        for i in trange(max_iters):
            trainer.train(0.001, 1)
            lc_arr_i, _  = ng_trainer.train(0.01, 1)
            lc_arr.append(float(lc_arr_i[0]))

    print(m.get_objective())
    m.print()

    plt.plot(lc_arr)
    plt.yscale('log')
    plt.show()


# predict

pred_mu, pred_var = m.predict_latents(XS)
pred_var = np.diagonal(pred_var, axis1=1, axis2=2)

plt.fill_between(
    np.squeeze(XS), 
    pred_mu[:, 0] - 1.96 * np.sqrt(pred_var[:, 0]),
    pred_mu[:, 0] + 1.96 * np.sqrt(pred_var[:, 0]),
    alpha = 0.4
)
plt.plot(XS, pred_mu[:, 0])
plt.scatter(X, Y[:, 0])
plt.show()

