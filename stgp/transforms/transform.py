"""
Base transform class.

Implements
    Transform: the base class of all transformations
    NonLinearTransform
    LinearTransform
    Joint
    Independent
"""
from ..core import Prior, GPPrior, Model
from ..utils.utils import ensure_module_list, can_batch, get_batch_type
from batchjax import batch_or_loop, BatchType
from ..computation.matrix_ops import to_block_diag, batched_diagonal_from_XDXT

import jax
import jax.numpy as np
import objax
import chex

from typing import List, Optional

class Transform(GPPrior):
    """
    All transforms must define:
        output_dim: number of latent gp outputs
        input_dim: number of input gps
        parent: pointer to parent gp / list of gps
        forward: the transformation
        base_prior: return the base prior

    Linear Transforms must implement
        transform(mu, var): transform an input gaussian

    When a transform defines a base prior it must additionally define
        get_sparsity_list
        get_Z
        get_sparsity
    
    """
    def __init__(self):
        self._output_dim = None
        self._input_dim = None

        # parent obj that is being transformed
        self._parent = None

    def transform_diagonal(self, mu, var):
        """ Transform a diagonal gaussian dist """
        raise NotImplementedError()

    def transform(self, mu, var):
        """ 
        Transform a full gaussian dist 

        mu and var are rank 2, should return rank 2.
        """
        
        raise NotImplementedError()

    def forward(self, x):
        """Compute f=T(x)."""
        raise NotImplementedError()

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        raise NotImplementedError()

    @property
    def is_base(self):
        return False

    @property
    def base_prior(self):
        """
        A transform is paced on top of a GP prior. This returns that base GP prior.
        """
        return self.parent.base_prior

    @property
    def num_outputs(self): raise RuntimeWarning('num_outputs has been removed. Use output_dim instead.')

    @property
    def output_dim(self): return self._output_dim

    @property
    def input_dim(self): return self._input_dim

    @property
    def parent(self): return self._parent

    def mean(self, XS): raise NotImplementedError()
    def mean_blocks(self, XS):raise NotImplementedError()
    def var(self, XS):raise NotImplementedError()
    def var_blocks(self, XS):raise NotImplementedError()
    def covar_blocks(self, X1, X2):raise NotImplementedError()
    def covar(self, X1, X2):raise NotImplementedError()
    def full_var(self, X):raise NotImplementedError()

class NonLinearTransform(Transform):
    def __init__(self, latent):
        self._parent = latent

class LinearTransform(Transform):
    def __init__(self, latent):
        self._parent = latent

class Joint(Transform):
    @property
    def is_base(self):
        return True

class Independent(Transform):
    def __init__(
        self, 
        latents: Optional[List['Model']] = None, 
        latent: Optional['Model'] = None,
        prior=True
    ) -> None:
        """
        Args:
            prior: bool -- Indicates whether latents are priors or posteriors
        """ 
        super().__init__()

        self.prior = prior

        if (latents is None) and (latent is None):
            raise RuntimeError('Latents must be passed')

        if (latent is not None) and (latents is not None):
            raise RuntimeError('Only latent or latents must be passed')

        if latent:
            # Standardize input to ease implementation
            latents = [latent]

        if prior:
            self._output_dim = len(latents)
        else:
            self._output_dim = latent.output_dim

        self._input_dim = self.output_dim
        self._parent = ensure_module_list(latents)

    @property
    def latents(self):
        return self.parent

    @property
    def is_base(self):
        return True

    @property
    def base_prior(self):
        return self

    def get_sparsity_list(self):
        return [p.sparsity for p in self.parent]

    def get_Z(self):
        Z_arr = batch_or_loop(
            lambda latent:  latent.get_Z(),
            [self.parent],
            [0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        return Z_arr

    def mean_blocks(self, X1: np.ndarray) -> np.ndarray:
        mean = batch_or_loop(
            lambda X1, latent:  latent.mean(X1),
            [X1, self.parent],
            [None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        mean = np.reshape(
            mean, 
            [self.output_dim, X1.shape[0], 1]
        )

        return mean

    def mean(self, X1):
        return np.vstack(self.mean_blocks(X1))

    def covar_blocks(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        k_arr = batch_or_loop(
            lambda X1, X2, latent:  latent.covar(X1, X2),
            [X1, X2, self.parent],
            [None, None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        k_arr = np.reshape(
            k_arr, 
            [self.output_dim, X1.shape[0], X2.shape[0]]
        )

        return k_arr

    def covar(self, X1, X2):
        c_blocks = self.covar_blocks(X1, X2)
        return to_block_diag(c_blocks)


    def var_blocks(self, X1: np.ndarray) -> np.ndarray:
        var = batch_or_loop(
            lambda X1, latent:  latent.var(X1),
            [X1, self.parent],
            [None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        var = np.reshape(
            var, 
            [self.output_dim, X1.shape[0], 1]
        )

        return var

    def var(self, X):
        v_blocks = self.var_blocks(X)
        v_stacked =  np.vstack(v_blocks)
        chex.assert_shape(v_stacked, [X.shape[0]*self.output_dim, 1])
        return v_stacked

    def full_var_blocks(self, X1: np.ndarray) -> np.ndarray:
        return self.covar_blocks(X1, X1)

    def state_space_representation(self, X_s):
        F_blocks, L_blocks, Qc_blocks, H_blocks, P_inf_blocks = batch_or_loop(
            lambda x_s, latent:  latent.kernel.to_ss(x_s),
            [X_s, self.parent],
            [None, 0],
            dim = self.output_dim,
            out_dim = 5,
            batch_type = get_batch_type(self.parent)
        )

        F = to_block_diag(F_blocks)
        L = to_block_diag(L_blocks)
        Qc = to_block_diag(Qc_blocks)
        P_inf = to_block_diag(P_inf_blocks)
        H = to_block_diag(H_blocks)

        return F, L, Qc, H, P_inf



class MultiOutput(Transform):
    def __init__(self, parent):
        # all objects in parent must share the SAME base prior
        self._parent = objax.ModuleList(
            parent
        )

        self._output_dim = sum([p.output_dim for p in self.parent])

    def transform(self, mu, var):
        mu_arr = []
        var_arr = []

        # each must return a single output
        for p in self.parent:
            _m, _v = p.transform(mu, var)
            mu_arr.append(_m)
            var_arr.append(_v)

        # TODO: this assuming a single output but if we return the list here we can generalise this
        return np.vstack(mu_arr), to_block_diag(var_arr)

    def forward(self, f):
        res = []
        for p in self.parent:
            res.append(
                p.forward(f)
            )
        return np.hstack(res)

    @property
    def base_prior(self):
        # all objects in parent share the same base_prior so we can just return the first one
        return self.parent[0].base_prior


