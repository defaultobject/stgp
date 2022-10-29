from . import Kernel
import jax
import jax.numpy as np
from jax import jacfwd, jacrev, grad
from ..computation.matrix_ops import hessian

import chex

class DerivativeKernel(Kernel):
    """
    Accecpts a parent kernel OR a parent model
    """
    def __init__(self, parent_kernel = None) :
        self.parent_kernel = parent_kernel
        self.active_dims = None

    def _K(self, X1, X2):
        """ This is is being used a prior kernel therefore we can pass through the
        kernel function """

        return self._K_from_fn(X1, X2, self.parent_kernel.K)

    def K_from_fn(self, X1, X2, var_fn):
        return self._K_from_fn(X1, X2, var_fn)

class FirstOrderDerivativeKernel(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None,
            input_index: int = 0
        ):
        """
        Args:
            input_idx (int): which input dimension of X to take derivates of
        """

        super(FirstOrderDerivativeKernel, self).__init__(parent_kernel)
        self.output_dim = 2
        self.input_index = input_index

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t] then we use 
            Tto denote the differential operators d/dt

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2)             
            (T)K,    (T)K(T),    (T)K(T^2)        
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2)   

        """
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # K(T)
        res01 = grad(k, argnums=(1))(x1, x2)


        # Computes
        # (T)K(T)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)


        # Construct full matrix
        # K,       K(T))
        # (T)K,    (T)K(T)

        K = np.array([
            [res00,       res01[self.input_index]], # f
            [res10[self.input_index],    res11[self.input_index, self.input_index]], # df/dt
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1]],
            [K[:, :, 1, 0], K[:, :, 1, 1]],
        ])

        return K_reshaped

class SecondOrderDerivativeKernel(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None,
            input_index: int = 0
        ):

        super(SecondOrderDerivativeKernel, self).__init__(parent_kernel)
        self.output_dim = 3
        self.input_index = input_index

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t] then we use 
            Tto denote the differential operators d/dt

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2)             
            (T)K,    (T)K(T),    (T)K(T^2)        
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2)   

        """
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # K(T)
        res01 = grad(k, argnums=(1))(x1, x2)

        # (T^2)K
        res20 = hessian(k, argnums=(0))(x1, x2)

        # K(T^2)
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2)
        res12 = hessian(grad(k, argnums=(0)), argnums=(1))(x1, x2)
        # (T^2)K(T)
        res21 = hessian(grad(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        # (T^2)K(T^2)
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(T^2)
        # (T)K,    (T)K(T),    (T)K(T^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2)

        K = np.array([
            [res00,       res01[self.input_index],        res02[self.input_index, self.input_index]], # f
            [res10[self.input_index],    res11[self.input_index, self.input_index],     res12[self.input_index, self.input_index, self.input_index]], # df/dt
            [res20[self.input_index][self.input_index], res21[self.input_index, self.input_index, self.input_index],  res22[self.input_index, self.input_index, self.input_index, self.input_index]], # d^2f/dt^2
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2]],
        ])

        return K_reshaped




class FirstOrderDerivativeKernel_1D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None,
        ):

        super(FirstOrderDerivativeKernel_1D, self).__init__(parent_kernel)
        self.output_dim = 2

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t] then we use 
            Tto denote the differential operators d/dt

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2)             
            (T)K,    (T)K(T),    (T)K(T^2)        
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2)   

        """
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # K(T)
        res01 = grad(k, argnums=(1))(x1, x2)


        # Computes
        # (T)K(T)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)


        # Construct full matrix
        # K,       K(T))
        # (T)K,    (T)K(T)

        K = np.array([
            [res00,       res01[0]], # f
            [res10[0],    res11[0, 0]], # df/dt
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1]],
            [K[:, :, 1, 0], K[:, :, 1, 1]],
        ])

        return K_reshaped


class SecondOrderDerivativeKernel_1D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None,
        ):

        super(SecondOrderDerivativeKernel_1D, self).__init__(parent_kernel)
        self.output_dim = 3

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t] then we use 
            Tto denote the differential operators d/dt

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2)             
            (T)K,    (T)K(T),    (T)K(T^2)        
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2)   

        """
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # K(T)
        res01 = grad(k, argnums=(1))(x1, x2)

        # (T^2)K
        res20 = hessian(k, argnums=(0))(x1, x2)

        # K(T^2)
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2)
        res12 = hessian(grad(k, argnums=(0)), argnums=(1))(x1, x2)
        # (T^2)K(T)
        res21 = hessian(grad(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        # (T^2)K(T^2)
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(T^2)
        # (T)K,    (T)K(T),    (T)K(T^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2)

        K = np.array([
            [res00,       res01[0],        res02[0, 0]], # f
            [res10[0],    res11[0, 0],     res12[0, 0, 0]], # df/dt
            [res20[0][0], res21[0, 0, 0],  res22[0, 0, 0, 0]], # d^2f/dt^2
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2]],
        ])

        return K_reshaped


class SecondOrderDerivativeKernel_2D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(SecondOrderDerivativeKernel_2D, self).__init__(parent_kernel)
        self.output_dim = 5

    def _compute_derivatives(self, x1, x2, var_fn):
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
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

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

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

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

class SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D(SecondOrderDerivativeKernel_2D):
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D, self).__init__(parent_kernel)
        self.output_dim = 3

    def _K_from_fn(self, X1, X2, var_fn):
        Kxx = var_fn(X1, X2)

        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0],  K[:, :, 0, 1], K[:, :, 0, 4]],
            [K[:, :, 1, 0],  K[:, :, 1, 1], K[:, :, 1, 4]],
            [K[:, :, 4, 0],  K[:, :, 4, 1], K[:, :, 4, 4]]
        ])

        return K_reshaped


class SecondOrderDerivativeKernel_3D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(SecondOrderDerivativeKernel_3D, self).__init__(parent_kernel)
        self.output_dim = 7

    def _compute_derivatives(self, x1, x2, var_fn):
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
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]


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

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

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

