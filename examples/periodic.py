import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp
import objax

import numpy as np
import legogp as lego
from legogp.transforms import Independent
from legogp.transforms.latent_force import NonLinearLFM, LotkaVolterra, Linearized, PopulationLotkaVolterra, RM_Population, LatentForce
from legogp.transforms.sdes import LTI_SDE, EulerMaruyama
from legogp.kernels import Matern32, ScaledMatern32, Periodic, ApproxSDEPeriodic
from legogp.models import GP
from legogp.sparsity import NoSparsity
from legogp.computation.solvers.euler import euler
from legogp.computation.filters import kalman_filter as kf
from legogp.computation.filters import rts_smoother as rts
from legogp.inference import StatisticallyLinearisedFilter
from legogp.data import Data, MultiOutputTemporalData, TemporalData
from legogp import settings
from legogp.core import Model
from legogp.likelihood import Gaussian, BlockDiagonalGaussian, ReshapedGaussian
from legogp import Parameter

from legogp.computation.parameter_transforms import inv_probit, probit

from legogp.trainers import GradDescentTrainer, ScipyTrainer
from legogp.trainers.callbacks import progress_bar_callback

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(0)

N = 100
x = np.linspace(0, 1, N)
y = np.sin(x*10) + 1e-3 * np.random.randn(N)
X = x[:, None]
Y = y[:, None]
XS = np.linspace(-1, 2, 1000)[:, None]

lego.settings.jitter = 1e-5


if False:
    data = TemporalData(X=X, Y=Y)
    kern = ApproxSDEPeriodic(1.0, 1.0, 1.0, n_terms=1)

    likelihood = BlockDiagonalGaussian(1, 1, variance=1e-1*np.tile(np.ones([1, 1]), [1, 1, 1]))
    likelihood = ReshapedGaussian(likelihood, data.Nt, 1)

    latents = [
        GP(sparsity=NoSparsity(), kernel=kern) for q in range(1)
    ]

    base_gp = LTI_SDE(Independent(latents))

    m = GP(
        data = data,
        likelihood = likelihood,
        prior = base_gp,
        inference = 'Sequential'
    )
else:
    data = Data(X=X, Y=Y)
    kern = Periodic(1.0, 1.0, 1.0)

    likelihood = Gaussian(0.01)

    m = GP(
        data = data,
        likelihood = likelihood,
        kernel = kern
    )

print(m.get_objective())

if False:
    trainer = ScipyTrainer(m, 'L-BFGS-B')
    epochs = 100
    callback = progress_bar_callback(epochs)
    learning_rates, _ = trainer.train(0.01, epochs, callback=callback)

else:
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)
    epochs = 100
    callback = progress_bar_callback(epochs)
    learning_rates, _ = trainer.train(0.01, epochs, callback=callback)

plt.plot(learning_rates)
plt.show()

pred_mu, pred_var = m.predict_f(XS)

plt.plot(XS, pred_mu)
plt.scatter(X, Y)
plt.show()
