import jax 
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as np
import numpy as onp
from jax import jacfwd, jacrev, grad

import stgp
from stgp.kernels import RBF, Kernel
from stgp.transforms import LinearTransform, Independent, DataLatentPermutation, DataLatentPermutationFromFull
from stgp.models import GP
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
from stgp.data import Data


import stdata as st

import matplotlib.pyplot as plt

from typing import List

X_3d = st.grids.create_spatial_temporal_grid(
    onp.linspace(0, 1, 10), 0, 1, 0, 1, 10, 10
)   

X_2d = st.grids.create_spatial_grid(
    0, 1, 0, 1, 10, 10
) 

Y_2d = (X_2d[:, 0] + onp.random.randn(X_2d.shape[0]))[:, None]

class DifferentialOperatorJoint(LinearTransform):
    def __init__(
        self,
        base_latent,
        derivative_kernel
    ):
        if base_latent is None:
            raise RuntimeError('Latent gp must be passed')

        self._latent_obj = base_latent
        self.derivative_kernel = derivative_kernel
        self._output_dim = self.derivative_kernel.output_dim
        self._latents_arr = [self.latent_obj]

    def mean(self, X1):
        return np.zeros([X1.shape[0] * self.output_dim, 1])
    def covar(self, X1, X2):
        return self.derivative_kernel.K(X1, X2)

def hessian(f, argnums):
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)

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


class SecondOrderDerivativeKernel_3D(Kernel):
    def __init__(
            self, 
            base_kernel
        ):

        self.base_kernel = base_kernel
        self.active_dims = None
        self.output_dim = 7

    def _compute_derivatives(self, x1, x2):
        """
        Let x1 have columns denotes by [t, s1, s2] then we use 
            T, S1, S2 to denote the differential operators d/dt, d/ds1, d/ds2 

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2),       K(S1),       K(S1^2),       K(S2),       K(S2^2)
            (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2),    (T)K(S2),    (T)K(S2^2)
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2),  (T)^2K(S2),  (T)^2K(S2^2)
            (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2),   (S1)K(S2),   (S1)K(S2^2)
            (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2), (S1^2)K(S2), (S1^2)K(S2^2)
            (S2)K,   (S2)K(T),   (S2)K(T^2),   (S2)K(S1),   (S2)K(S1^2),   (S2)K(S2),   (S2)K(S2^2)
            (S2)^2K, (S2)^2K(T), (S2)^2K(T^2), (S2)^2K(S1), (S2)^2K(S1^2), (S2)^2K(S2), (S2)^2K(S2^2)

        """
        # fix shapes
        k = lambda x1, x2: self.base_kernel.K(x1[None, ...], x2[None, ...])[0, 0]


        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K, (S1)K, (S2)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # [K(T), K(S1), K(S2)]^T
        res01 = grad(k, argnums=(1))(x1, x2)

        # (T^2)K, (T)(S1)K, (T)(S2)K
        # (T)(S1)K, (S1^2)K, (S1)(S2)K
        # (T)(S2)K, (S1)(S2)K, (S2^2)K
        res20 = hessian(k, argnums=(0))(x1, x2)

        # K(T^2), K(T)(S1), K(T)(S2)
        # K(T)(S1), K(S1^2), K(S1)(S2)
        # K(T)(S2), K(S1)(S2), K(S2^2)
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T), (T)K(S1), (T)K(S2)
        # (S1)K(T), (S1)K(S1), (S1)K(S2)
        # (S2)K(T)(S2), (S2)K(S1), (S2)K(S2)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2),   (T)K(T)(S1),  (T)K(T)(S2)
        # (T)K(T)(S1), (T)K(S1^2),   (T)K(S1)(S2)
        # (T)K(T)(S2), (T)K(S1)(S2), (T)K(S2^2)
        #-
        # (S1)K(T^2),   (S1)K(T)(S1),  (S1)K(T)(S2)
        # (S1)K(T)(S1), (S1)K(S1^2),   (S1)K(S1)(S2)
        # (S1)K(T)(S2), (S1)K(S1)(S2), (S1)K(S2^2)
        #-
        # (S2)K(T^2),   (S2)K(T)(S1),  (S2)K(T)(S2)
        # (S2)K(T)(S1), (S2)K(S1^2),   (S2)K(S1)(S2)
        # (S2)K(T)(S2), (S2)K(S1)(S2), (S2)K(S2^2)
        res12 = hessian(grad(k, argnums=(0)), argnums=(1))(x1, x2)
        res21 = hessian(grad(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(T^2),       K(S1),       K(S1^2),       K(S2),       K(S2^2)
        # (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2),    (T)K(S2),    (T)K(S2^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2),  (T)^2K(S2),  (T)^2K(S2^2)
        # (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2),   (S1)K(S2),   (S1)K(S2^2)
        # (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2), (S1^2)K(S2), (S1^2)K(S2^2)
        # (S2)K,   (S2)K(T),   (S2)K(T^2),   (S2)K(S1),   (S2)K(S1^2),   (S2)K(S2),   (S2)K(S2^2)
        # (S2)^2K, (S2)^2K(T), (S2)^2K(T^2), (S2)^2K(S1), (S2)^2K(S1^2), (S2)^2K(S2), (S2)^2K(S2^2)

        K = np.array([
            [res00,       res01[0],        res02[0, 0],       res01[1],       res02[1, 1],       res01[2],       res02[2, 2]], # f
            [res10[0],    res11[0, 0],     res12[0, 0, 0],    res11[0, 1],    res12[0, 1, 1],    res11[0, 2],    res12[0, 2, 2]], # df/dt
            [res20[0][0], res21[0, 0, 0],  res22[0, 0, 0, 0], res21[1, 0, 0], res22[0, 0, 1, 1], res21[2, 0, 0], res22[0, 0, 2, 2]], # d^2f/dt^2
            [res10[1],    res11[1, 0],     res12[1, 0, 0],    res11[1, 1],    res12[1, 1, 1],    res11[1, 2],    res12[1, 2, 2]], # df / dx1
            [res20[1][1], res21[0, 1, 1],  res22[0, 0, 1, 1], res21[1, 1,1],  res22[1, 1, 1, 1], res21[2, 1, 1], res22[1, 1, 2, 2]],# d^2f / dx1^2
            [res10[2],    res11[2, 0],     res12[2, 0, 0],    res11[2,1],     res12[2, 1, 1],    res11[2, 2],    res12[2, 2, 2]],# df / dx2
            [res20[2][2], res21[0, 2, 2],  res22[0, 0, 2, 2], res21[1, 2, 2], res22[2, 2, 1, 1], res21[2, 2, 2], res22[2, 2, 2, 2]] # d^2f / dx2^2
        ])

        return K

    def _K(self, X1, X2):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0))(x1, X2)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2], K[:, :, 0, 3], K[:, :, 0, 4], K[:, :, 0, 5], K[:, :, 0, 6]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2], K[:, :, 1, 3], K[:, :, 1, 4], K[:, :, 1, 5], K[:, :, 1, 6]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2], K[:, :, 2, 3], K[:, :, 2, 4], K[:, :, 2, 5], K[:, :, 2, 6]],
            [K[:, :, 3, 0], K[:, :, 3, 1], K[:, :, 3, 2], K[:, :, 3, 3], K[:, :, 3, 4], K[:, :, 3, 5], K[:, :, 3, 6]],
            [K[:, :, 4, 0], K[:, :, 4, 1], K[:, :, 4, 2], K[:, :, 4, 3], K[:, :, 4, 4], K[:, :, 4, 5], K[:, :, 4, 6]],
            [K[:, :, 5, 0], K[:, :, 5, 1], K[:, :, 5, 2], K[:, :, 5, 3], K[:, :, 5, 4], K[:, :, 5, 5], K[:, :, 5, 6]],
            [K[:, :, 6, 0], K[:, :, 6, 1], K[:, :, 6, 2], K[:, :, 6, 3], K[:, :, 6, 4], K[:, :, 6, 5], K[:, :, 6, 6]],
        ])

        return K_reshaped

if False:
    base_kernel_3d = RBF(input_dim = 3, lengthscales = [1.0, 1.0, 1.0])
    pinn_kernel_3d = SecondOrderDerivativeKernel_3D(base_kernel_3d)

    K_3d = pinn_kernel_3d.K(X_3d, X_3d)

    plt.imshow(K_3d)
    plt.show()
else:
    data = Data(X_2d, Y_2d)
    base_kernel_2d = RBF(input_dim = 2, lengthscales = [1.0, 1.0])

    base_gp = GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X_2d), 
        kernel = base_kernel_2d
    )

    pinn_kernel_2d = SecondOrderDerivativeKernel_2D(base_kernel_2d)

    prior = DifferentialOperatorJoint(
        base_gp,
        pinn_kernel_2d
    )

    lik = stgp.likelihood.Gaussian(1.0)

    prior = DataLatentPermutationFromFull(prior)

    m = GP(
        data = data,
        likelihood = lik,
        prior = prior,
        inference='Variational',
        approximate_posterior = FullGaussianApproximatePosterior(
            dim = X_2d.shape[0]*pinn_kernel_2d.output_dim
        )
    )
    print(X_2d.shape)
    print(Y_2d.shape)
    print(m.get_objective())

    breakpoint()

    K_2d = pinn_kernel_2d.K(X_2d, X_2d)

    plt.imshow(K_2d)
    plt.show()

