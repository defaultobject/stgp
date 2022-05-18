import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp
import objax

import numpy as np
import legogp as lego
from legogp.transforms import Independent
from legogp.transforms.latent_force import NonLinearLFM, LotkaVolterra, Linearized, PopulationLotkaVolterra, RM_Population
from legogp.transforms.sdes import LTI_SDE, EulerMaruyama
from legogp.kernels import Matern32, ScaledMatern32
from legogp.models import GP
from legogp.sparsity import NoSparsity
from legogp.computation.solvers.euler import euler
from legogp.computation.filters import kalman_filter as kf
from legogp.computation.filters import rts_smoother as rts
from legogp.inference import StatisticallyLinearisedFilter
from legogp.data import Data, MultiOutputTemporalData
from legogp import settings
from legogp.core import Model
from legogp.likelihood import Gaussian, BlockDiagonalGaussian, ReshapedGaussian

from legogp.trainers import GradDescentTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback

from pathlib import Path
import pandas as pd

import mpl_interactions.ipyplot as iplt
import matplotlib.pyplot as plt

def get_data():
    data_path = Path('/Users/ohamelijnck/Documents/projects/turing-HY/code/datasets/lynx_hare/downloaded_data')
    df = pd.read_csv(data_path / 'lynx_hare.csv')
    return df

df = get_data()

if False:
    plt.plot(df['year'], df['lynx'], label='lynx')
    plt.plot(df['year'], df['hare'], label='hare')
    plt.legend()
    plt.show()

X = np.array(df['year'][:, None])
Y = np.array(df[['lynx', 'hare']])

X = X - np.min(X)

Y = np.log(Y) 

X_vis = np.copy(X)
Y_vis = np.copy(Y)

Y[40:60, 1] = np.NaN

XS = np.linspace(np.min(X), np.max(X), 10000)[:, None]
YS = np.ones([XS.shape[0], 2])*np.NaN

X = np.vstack([X, XS])
Y = np.vstack([Y, YS])


data = MultiOutputTemporalData(X=X, Y=Y)

latents = [
    GP(sparsity=NoSparsity(), kernel=ScaledMatern32(lengthscales=[10.0], variance=1.0))
    for q in range(2)
]


likelihood = BlockDiagonalGaussian(2, 1, variance=1e-2*np.tile(np.eye(2), [1, 1, 1]))
likelihood = ReshapedGaussian(likelihood, data.Nt, 2)

base_gp = LTI_SDE(Independent(latents))

a = 3.2 
b = 0.6
c= 50.0
d= 0.56
k = 125.0
r = 1.6


alpha = -0.57
beta = -0.23
delta = -0.2
gamma = -0.17

#m_lfm = PopulationLotkaVolterra(base_gp, 3.2, 0.6, 50.0, 0.56, 125.0, 1.6, init_state = Y[0, :])
#m_lfm = LotkaVolterra(base_gp, alpha, beta, delta, gamma, init_state = Y[0, :])


alpha = 0.9
K = 100
beta = 1.0
b = 10.0
gamma = 1.0
delta = 0.3

fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

m_lfm = RM_Population(base_gp, alpha, K, beta, b, gamma, delta, init_state = np.array([2.0, 1.0]))
res, init_x = euler(m_lfm, np.array([2.0, 1.0]), 100000, 0.001)

x = np.cumsum(np.ones(100000) * 0.001)

def _f1(*args, **kwargs):
    breakpoint()

def f1(x, a, K, beta, b, gamma, delta):
    alpha = a
    m_lfm = RM_Population(base_gp, alpha, K, beta, b, gamma, delta, init_state = np.array([2.0, 1.0]))
    res, x = euler(m_lfm, np.array([2.0, 1.0]), 100000, 0.001)
    return x[:, 0]

def f2(x, a, K, beta, b, gamma, delta):
    alpha = a
    m_lfm = RM_Population(base_gp, alpha, K, beta, b, gamma, delta, init_state = np.array([2.0, 1.0]))
    res, x = euler(m_lfm, np.array([2.0, 1.0]), 100000, 0.001)
    return x[:, 1]

controls = iplt.plot(x, f1, a=(1e-5, 10, 1000), K=(1e-5, 100, 1000), beta=(1e-5, 10, 1000), b=(1e-5, 10, 1000), gamma=(1e-5, 20, 1000), delta=(1e-5, 20, 1000), label="f1")
controls.params = {
    'a': alpha,
    'K': K,
    'beta': beta,
    'b': b,
    'gamma': gamma,
    'delta': delta,
}
iplt.plot(x, f2, controls=controls, label="f2")

plt.scatter(X[:, 0], Y[:, 0])
plt.scatter(X[:, 0], Y[:, 1])

_ = plt.legend()
plt.show()


