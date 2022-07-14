""" Single GP regression """
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp.trainers import GradDescentTrainer, ScipyTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms import LinearTransform
from stgp.transforms.basic import InputMeanFunction
from stgp.core.model_types import get_model_type
from stgp.transforms.pdes import HeatEquation2D

import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import matplotlib.pyplot as plt
from pathlib import Path


class SecondOrderDerivativeKernel_2D(Kernel):
    def __init__(
            self, 
            base_kernel
        ):

        self.base_kernel = base_kernel
        self.active_dims = None
        self.output_dim = 5

    def _compute_derivatives(self, x1, x2):
        """
        Let x1 have columns denotes by [t, s1] then we use 
            T, S1 to denote the differential operators d/dt, d/ds1

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2),       K(S1),       K(S1^2),      
            (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2),    
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2),  
            (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2),   
            (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2), 

        """
        # fix shapes
        k = lambda x1, x2: self.base_kernel.K(x1[None, ...], x2[None, ...])[0, 0]


        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K, (S1)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # [K(T), K(S1)]^T
        res01 = grad(k, argnums=(1))(x1, x2)

        # (T^2)K, (T)(S1)K
        # (T)(S1)K, (S1^2)K
        res20 = hessian(k, argnums=(0))(x1, x2)

        # K(T^2), K(T)(S1)
        # K(T)(S1), K(S1^2)
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T), (T)K(S1)
        # (S1)K(T), (S1)K(S1)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2),   (T)K(T)(S1)
        # (T)K(T)(S1), (T)K(S1^2)
        #-
        # (S1)K(T^2),   (S1)K(T)(S1)
        # (S1)K(T)(S1), (S1)K(S1^2)
        #-
        # (S2)K(T^2),   (S2)K(T)(S1)
        # (S2)K(T)(S1), (S2)K(S1^2)
        res12 = hessian(grad(k, argnums=(0)), argnums=(1))(x1, x2)
        res21 = hessian(grad(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)


        # Construct full matrix
        # K,       K(T),       K(T^2),       K(S1),       K(S1^2)
        # (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2)
        # (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2)
        # (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2)

        K = np.array([
            [res00,       res01[0],        res02[0, 0],       res01[1],       res02[1, 1]], # f
            [res10[0],    res11[0, 0],     res12[0, 0, 0],    res11[0, 1],    res12[0, 1, 1]], # df/dt
            [res20[0][0], res21[0, 0, 0],  res22[0, 0, 0, 0], res21[1, 0, 0], res22[0, 0, 1, 1]], # d^2f/dt^2
            [res10[1],    res11[1, 0],     res12[1, 0, 0],    res11[1, 1],    res12[1, 1, 1]], # df / dx1
            [res20[1][1], res21[0, 1, 1],  res22[0, 0, 1, 1], res21[1, 1,1],  res22[1, 1, 1, 1]]# d^2f / dx1^2
        ])

        return K

    def _K(self, X1, X2):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0))(x1, X2)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2], K[:, :, 0, 3], K[:, :, 0, 4]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2], K[:, :, 1, 3], K[:, :, 1, 4]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2], K[:, :, 2, 3], K[:, :, 2, 4]],
            [K[:, :, 3, 0], K[:, :, 3, 1], K[:, :, 3, 2], K[:, :, 3, 3], K[:, :, 3, 4]],
            [K[:, :, 4, 0], K[:, :, 4, 1], K[:, :, 4, 2], K[:, :, 4, 3], K[:, :, 4, 4]]
        ])

        return K_reshaped


# Fix randomness
np.random.seed(0)

# Generate Data
N = 100

x = np.linspace(0, 1, N)
y = np.sin(x*10) + 0.1*np.random.randn(N)
X = x[:, None]
Y = y[:, None]

D = 1

XS = np.linspace(-1, 2, 1000)[:, None]

data = stgp.data.Data(X, Y)

base_kernel_2d = RBF(input_dim = 2, lengthscales = [1.0, 1.0])

base_gp = GP(
    sparsity=stgp.sparsity.NoSparsity(Z=X), 
    kernel = base_kernel_2d
)

pinn_kernel_2d = SecondOrderDerivativeKernel_2D(base_kernel_2d)

prior = DifferentialOperatorJoint(
    base_gp,
    pinn_kernel_2d
)

prior = HeatEquation2D(prior)
breakpoint()

exit()


latent_gp = GP(
    sparsity= stgp.sparsity.NoSparsity(Z_ref=data._X), 
    kernel = ScaleKernel(RBF(input_dim=D, lengthscales=[1.0 for d in range(D)]))
)

latent_gp = InputMeanFunction(latent_gp)


# Create Model
m = stgp.models.GP(
    data=data,
    prior=latent_gp,
    likelihood = Gaussian(variance=0.1)
)

print(m.get_objective())

