import jax
import jax.numpy as np
import chex

from ..core import Block 
from . import Transform, LinearTransform, Joint


from ..dispatch import evoke
from ..computation.matrix_ops import get_block_diagonal
from .. import Parameter
from ..computation.matrix_ops import hessian
from jax import jacfwd, jacrev, grad

class DifferentialOperatorJoint(LinearTransform, Joint):
    """
    tbd.
    """
    def __init__(
        self,
        base_latent,
        mean = None,
        kernel = None,
        is_base:bool = True,
        has_parent:bool = False,
        hierarchical = False,
        whiten_space=False
    ):
        if base_latent is None:
            raise RuntimeError('Latent gp must be passed')

        self._parent = base_latent
        self.derivative_mean = mean
        self.derivative_kernel = kernel
        self._output_dim = self.derivative_kernel.output_dim
        self._input_dim = 1
        self._is_base = is_base
        self.has_parent = has_parent
        self.hierarchical = hierarchical
        self.whiten_space = whiten_space

    @property
    def full_transform(self):
        # do not use batched transform
        return True

    @property
    def is_base(self):
        return self._is_base

    @property
    def out_block_dim(self):
        return self.derivative_kernel.output_dim

    @property
    def in_block_dim(self):
        # always requires the full input covariance to compute derivates
        return Block.FULL

    @property
    def in_block_type(self):
        # always requires the full input covariance to compute derivates
        return Block.FULL

    def forward(self, f): return f

    def transform(self, mu, var, data):
        if not self.hierarchical:
            # base prior so no need to transform
            chex.assert_rank([mu, var], [2, 2])
        else:
            # compute 
            # no time
            chex.assert_rank([mu, var], [3, 4])
            mu, var = evoke('spatial_conditional', data, self)(
                data, 
                self.parent.parent.sparsity.raw_Z, 
                mu, 
                var[:, 0, ...], 
                self
            )

            chex.assert_rank([mu, var], [3, 4])
        return mu, var

    def get_sparsity_list(self):
        return [self.get_sparsity()]

    def get_Z_stacked(self):
        Z = self.get_Z_blocks()
        Z = Z[None, ...]
        chex.assert_rank(Z, 4)
        return Z

    def get_Z_blocks(self):
        Z = self.parent.get_Z()
        Z_arr = np.tile(Z, [self.output_dim, 1, 1])

        chex.assert_rank(Z_arr, 3)
        return Z_arr

    def get_Z(self):
        Z_all = self.get_Z_blocks()

        Z = np.vstack(Z_all)
        chex.assert_rank(Z, 2)
        return Z

    def get_sparsity(self):
        if self.has_parent:
            return self.parent.get_sparsity()
        else:
            return self.parent.sparsity

    def mean(self, X1):
        if not self.has_parent:
            return np.zeros([X1.shape[0] * self.output_dim, 1])
        else:
            mean_fn = self.parent.mean_blocks
            mean_x = self.derivative_mean.mean_from_fn(X1, mean_fn)
            return mean_x

    def mean_blocks(self, X1):
        if not self.has_parent:
            return np.zeros([self.output_dim, X1.shape[0], 1])
        else:
            mean_fn = self.parent.mean_blocks
            mean_x = self.derivative_mean.mean_blocks_from_fn(X1, mean_fn)
            return mean_x

    def b_mean(self, X1):
        # for compatability with Independent X1 and can either be of rank 3 or 4
        if len(X1.shape) == 4 :
            chex.assert_equal([X1.shape[0]], [1])
            X1 = X1[0]

        # We only support zero mean so we simply return the mean
        return self.mean(X1[0])

    def b_mean_blocks(self, X1):
        return self.mean_blocks(X1[0])

    def covar_from_fn(self, X1, X2, var_fn):
        K_xz = self.derivative_kernel.K_from_fn(X1, X2, var_fn)
        return K_xz

    def covar(self, X1, X2):
        # if self does not have a parent then 
        #  we know that the prior covariance is directly given by K
        #  so we can efficient computed the derivate kernel
        if not self.has_parent:
            return self.derivative_kernel.K(X1, X2)
        else:
            var_fn = self.parent.covar

            return self.covar_from_fn(X1, X2, var_fn)

    def b_covar(self, X1, X2):
        """ WARNING: we assume that X1, X2 is actually repeated """

        # for compatability with Independent X1 and can either be of rank 3 or 4
        if len(X1.shape) == 4 and  len(X2.shape) == 4:
            chex.assert_equal([X1.shape[0], X2.shape[0]], [1, 1])
            X1 = X1[0]
            X2 = X2[0]

        return self.covar(X1[0], X2[0])

    def covar_blocks(self, X1, X2):
        # TODO: this might need to be permutated 
        #raise NotImplementedError() 
        K_full = self.covar(X1, X2)

        return get_block_diagonal(
            K_full,
            self.output_dim,
        )

    def b_covar_blocks(self, X1, X2):
        """ WARNING: we assume that X1 is actually repeated """

        return self.covar_blocks(X1[0], X2[0])

    def full_var(self, X):
        return self.covar(X, X)

    def var(self, X):
        fn = jax.vmap(lambda p,x: p.covar(x[None, :], x[None, :]), [None, 0])
        res = fn(self, X)
        diag_vec = np.diagonal(res, axis1=1, axis2=2)
        res =  np.hstack(diag_vec.T)[:, None]

        chex.assert_rank(res, 2)
        return res

    def var_blocks(self, X):
        fn = jax.vmap(lambda p,x: p.covar(x[None, :], x[None, :]), [None, 0])
        res = fn(self, X)
        diag_vec = np.diagonal(res, axis1=1, axis2=2)

        return diag_vec.T[..., None]

    @property
    def base_prior(self):
        #if self.is_base:
        if not self.hierarchical:
            return self

        return self.parent.base_prior

    @property
    def hierarchical_base_prior(self):
        if self.hierarchical:
            return self

        # when not hierarchical this should have the same behavior as base prior
        return self.base_prior

class PDE(Transform):

    def jac(self, x, X_s, t):
        """Compute d (self.forward(x))(dx) """
        chex.assert_rank(x, 2)
        J =  jax.jacfwd(lambda _x: self.forward(_x, X_s, t))(x)[..., 0] 
        chex.assert_rank(J, 2)
        return J

    def H(self, x, X_s, t):
        return self.jac(x, X_s, t)


class IdentityPDE(PDE):
    def __init__(self, latent, m_init = None):
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim


        if m_init is None:
            m_init = np.zeros(self.input_dim)[:, None]

        self.m_init = np.array(m_init)

    def m_inf(self, x, X_s, t):
        return self.m_init

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def forward(self, f):
        """ 
        f is of shape 3 corresponding to f, ft
        """
        return f[0]

class SimpleODE(PDE):
    def __init__(self, latent, m_init = None):
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        if m_init is None:
            m_init = np.zeros(self.input_dim)[:, None]

        self.m_init = np.reshape(np.array(m_init), [np.array(m_init).shape[0], 1])
        self.pred_mode = False

    def m_inf(self, x, X_s, t):
        return self.m_init

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def forward(self, f, X_s, t):
        """ 
        f is of shape 3 corresponding to f, ft
        """
        return f[1]-2*t

    def H(self, x, X_s, t):
        if self.pred_mode:
            return np.array([1.0, 0.0])[None, :]
        else:
            return self.jac(x, X_s, t)


class Pendulum1D(PDE):
    def __init__(self, latent, g, l, train=True):
        """
        Latent must be a DifferentialOperatorJoint with a 1D differential kernel.

        Let x be the angle

        The  transform is:
            d^2 x/dt^2 + sin(x) = 0
        """
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

        self.g_param = Parameter(
            np.array(g), 
            constraint='positive', 
            name ='Pendulum1D/g', 
            train=train
        )

        self.l_param = Parameter(
            np.array(l), 
            constraint='positive', 
            name ='Pendulum1D/l', 
            train=train
        )

    def m_inf(self, x, X_s, t):
        return np.zeros(self.input_dim)[:, None]

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def forward(self, f):
        """ 
        f is of shape 3 corresponding to f, ft, ft2
        """
        t = f[0]
        dt2 = f[2]

        ls = self.g_param.value / self.l_param.value

        res = dt2 + ls * np.sin(t)

        return np.array([res])
class DampedPendulum1D(PDE):
    def __init__(self, latent, b, g, l, train=True):
        """
        Latent must be a DifferentialOperatorJoint with a 1D differential kernel.

        Let x be the angle

        The  transform is:
            d^2 x/dt^2 + sin(x) + d x/dt = 0
        """
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

        self.b_param = Parameter(
            np.array(b), 
            constraint='positive', 
            name ='Pendulum1D/b', 
            train=train
        )

        self.g_param = Parameter(
            np.array(g), 
            constraint='positive', 
            name ='Pendulum1D/g', 
            train=train
        )

        self.l_param = Parameter(
            np.array(l), 
            constraint='positive', 
            name ='Pendulum1D/l', 
            train=train
        )

    def _f(self, init_x, t):
        ls = self.g_param.value / self.l_param.value
        b = self.b_param.value

        t = init_x[0]
        dt = init_x[1]

        return np.array([
            dt,
            - ls * np.sin(t) - b * dt
        ])

    def forward(self, f):
        """ 
        f is of shape 3 corresponding to f, ft, ft2
        """
        t = f[0]
        dt = f[1]
        dt2 = f[2]

        ls = self.g_param.value / self.l_param.value
        b = self.b_param.value

        res = dt2 + ls * np.sin(t) + b * dt

        return np.array([res])

class SpatialDampedPendulum(PDE):
    def __init__(self, latent, b, g, l, train=True):
        """
        Latent must be a DifferentialOperatorJoint with a 1D differential kernel.

        Let x be the angle

        The  transform is:
            d^2 x/ds^2 + sin(x) + d x/ds = 0
        """
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

        self.b_param = Parameter(
            np.array(b), 
            constraint='positive', 
            name ='Pendulum1D/b', 
            train=train
        )

        self.g_param = Parameter(
            np.array(g), 
            constraint='positive', 
            name ='Pendulum1D/g', 
            train=train
        )

        self.l_param = Parameter(
            np.array(l), 
            constraint='positive', 
            name ='Pendulum1D/l', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to f, fs, fs2, dt, ...
        """
        t = f[0]
        ds = f[1]
        ds2 = f[2]

        ls = self.g_param.value / self.l_param.value
        b = self.b_param.value

        res = ds2 + ls * np.sin(t) + b * ds

        return np.array([res])



class HeatEquation2D(PDE, LinearTransform):
    def __init__(self, latent):
        """
        Latent must be a DifferentialOperatorJoint with a 2D differential kernel.

        The heat kernel transform is:
            df/dt - d^2f/dx^2 = 0
        """
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

    def forward(self, f):
        """ 
        f is of shape 5 corresponding to f, ft, ft2, fx, fx2
        """
        return np.array([f[1] - f[4]])

    def _transform_mean(self, mu):

        dt = mu[1]
        dx2 = mu[4]

        return np.array([dt - dx2])

    def _transform_covar(self, var):

        N1 = int(var.shape[0]/self.input_dim)
        N2 = int(var.shape[1]/self.input_dim)

        left_idx = np.arange(N1)
        right_idx = np.arange(N2)

        Kt = var[left_idx+N1*1, :]
        Kt = Kt[:, right_idx+N2*1]

        Kx2 = var[left_idx+N1*4, :]
        Kx2 = Kx2[:, right_idx+N2*4]

        Ktx2 = var[left_idx+N1*1, :]
        Ktx2 = Ktx2[:, right_idx+N2*4]

        Kx2t = var[left_idx+N1*4, :]
        Kx2t = Kx2t[:, right_idx+N2*1]

        K =  Kt + Kx2 - Ktx2 - Kx2t

        #ensure correct rank
        chex.assert_shape(K, [N1 * self.output_dim, N2 * self.output_dim]) 

        return K

    def transform(self, mu, var):
        return self._transform_mean(mu), self._transform_covar(var)


    def mean(self, X):
        # parent is a DifferentialOperatorJoint whose outputs are:
        #   f, ft, ft2, fx, fx2 
        parent_mean = self.parent.mean_blocks(X)

        return self._transform_mean(parent_mean)

    def covar(self, X1, X2):
        parent_covar = self.parent.covar(X1, X2)

        return self._transform_covar(parent_covar)

class AllenCahn(PDE):
    def __init__(self, latent, train=True):
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

    def _f(self, init_x, t):
        raise NotImplementedError()

    def forward(self, f):
        """ 
        f is of shape 3 corresponding to f, ft fx2
        """
        t = f[0]
        dt = f[1]
        dx2 = f[2]

        res = dt - 0.0001 * dx2 + 5 * (t**3) - 5 * t

        return np.array([res])


class _LorenzSystemX(PDE):
    def __init__(self, latent, sigma, train=True):

        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim


        self.sigma_param = Parameter(
            np.array(sigma), 
            name ='LorenzSystem/sigma', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to x, xt, y, yt, z, zt 
        """
        x, xt, y, yt, z, zt = f
        sigma = self.sigma_param.value

        return (xt - sigma * (y - x))[:, None]

class _LorenzSystemY(PDE):
    def __init__(self, latent, rho, train=True):
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.rho_param = Parameter(
            np.array(rho), 
            name ='LorenzSystem/rho', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to x, xt, y, yt, z, zt 
        """
        x, xt, y, yt, z, zt = f
        rho = self.rho_param.value

        return (yt - x * (rho - z) + y)[:, None]

class _LorenzSystemZ(PDE):

    def __init__(self, latent, beta, train=True):
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.beta_param = Parameter(
            np.array(beta), 
            name ='LorenzSystem/beta', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to x, xt, y, yt, z, zt 
        """
        x, xt, y, yt, z, zt = f
        beta = self.beta_param.value

        return (zt - x * y + beta * z)[:, None]


def LorenzSystem(latent, sigma, rho, beta, train=True):
    """
    THe lorenz stystem describes the following system of equations

        dx/dt = sig * (y-x)
        dy/dt = x * ( rho - z) - y
        dz/dt = x * y - beta * z
    """
    return [
        _LorenzSystemX(latent[0], sigma, train=train), 
        _LorenzSystemY(latent[1], rho, train=train), 
        _LorenzSystemZ(latent[2], beta, train=train), 
    ]


class _LotkaVolterraSystemX(PDE):

    def __init__(self, latent, alpha, beta, train=True):
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.alpha_param = Parameter(
            np.array(alpha), 
            name ='LotkaVolterraSystem/alpha', 
            train=train
        )

        self.beta_param = Parameter(
            np.array(beta), 
            name ='LotkaVolterraSystem/beta', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 4 corresponding to x, xt, y, yt
        """
        x, xt, y, yt = f
        alpha = self.alpha_param.value
        beta = self.beta_param.value

        return (xt - (alpha * x - beta * x * y))[:, None]

class _LotkaVolterraSystemY(PDE):

    def __init__(self, latent, delta, gamma, train=True):
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.delta_param = Parameter(
            np.array(delta), 
            name ='LotkaVolterraSystem/delta', 
            train=train
        )

        self.gamma_param = Parameter(
            np.array(gamma), 
            name ='LotkaVolterraSystem/gamma', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 4 corresponding to x, xt, y, yt
        """
        x, xt, y, yt = f
        delta = self.delta_param.value
        gamma = self.gamma_param.value

        return (yt - (delta * x * y  - gamma * y))[:, None]

def LotkaVolterraSystem(latent, alpha, beta, delta, gamma, train=True):
    """
    THe lorenz stystem describes the following system of equations

        dx/dt = alpha *x - beta *x * y
        dy/dt = delta * x * y  - gamma * y
    """
    return [
        _LotkaVolterraSystemX(latent[0], alpha, beta, train=train), 
        _LotkaVolterraSystemY(latent[1], delta, gamma, train=train), 
    ]
