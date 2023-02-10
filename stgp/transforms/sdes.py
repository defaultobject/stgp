from . import Transform, Independent
from ..computation.permutations import data_order_to_output_order

import jax.numpy as np

class SDE(Transform):

    @property
    def latents(self):
        return self.gp.latents

    @property
    def num_latents(self):
        return len(self.latents)

class LTI_SDE(SDE):
    def __init__(self, gp: 'Model'):
        self.gp = gp

    @property
    def temporal_output_dim(self):
        """
        Only return `f'
        """
        return 1

    @property
    def _output_dim(self):
        return 1

    @property
    def spatial_output_dim(self):
        # TODO: this is a bit hacky atm

        # if the spatial kernel is a derivate kernel this will get ignored by the filter and computed explicitely after smoothing.
        try:
            return self.gp.base_prior.parent[0].kernel.k2.output_dim
        except Exception as e:
            return 1

    def state_space_representation(self, X_s, dt, t):
        return self.gp.state_space_representation(X_s)

    def f(self, x, X_s, t):
        F, _, _, _, _ = self.gp.state_space_representation(X_s)

        return F @ x

    def L(self, x, X_s, t):
        _, L, _, _, _ = self.gp.state_space_representation(X_s)

        return L

    def Q(self, x, X_s, t):
        _, _, Q, _, _ = self.gp.state_space_representation(X_s)

        return Q

    def H(self, x, X_s, t):
        _, _, _, H, _ = self.gp.state_space_representation(X_s)

        return H

    def P_inf(self, x, X_s, t):
        _, _, _, _, Pinf = self.gp.state_space_representation(X_s)

        return Pinf

    def m_inf(self, x, X_s, t):
        _, _, _, _, Pinf = self.gp.state_space_representation(X_s)

        m_inf = np.zeros([Pinf.shape[0], 1])

        return m_inf

    def expm(self, X_s, t):
        return self.gp.expm(t, X_s)

class LTI_SDE_Full_State_Obs(LTI_SDE):
    def __init__(self, gp: 'Model', whiten_space=False):
        self.gp = gp
        self._state_space_dim = sum(self.gp.state_space_dim())
        self.whiten_space = whiten_space

    @property
    def temporal_output_dim(self):
        """ Returns the full state.  """
        return self._state_space_dim
        #return 3

    @property
    def _output_dim(self):
        return self.temporal_output_dim * self.spatial_output_dim

    def H(self, x, X_s, t):

        # Observe both f and df
        H_t = np.eye(self._state_space_dim)

        #When there are no spatial points there is no need to permute
        #as it will automatically be in time-latent format
        if X_s is None:
            return H_t

        # need to permute from latent-space-state to latent-state-space
        # need to permute from ds-space-df to df-ds-space
        # assuming that latent = 1 and we are treating state as latent

        # the latent-data is (diff_s x space) x diff_t
        # is (Ns) x P
        Ns = self.spatial_output_dim * X_s.shape[0]
        P = self.temporal_output_dim

        # the result will now be (diff_t x diff_s x space)
        H = data_order_to_output_order(P, Ns).T

        return H

class LTI_SDE_Full_State_Obs_With_Mask(LTI_SDE):
    """
    Observe partial deriatives. Useful when we have observations on [f, df/dt] but we want to use a smoother kernel like the matern52/72 etc.
    """
    def __init__(self, gp: 'Model', keep_dims, whiten_space=False):
        self.gp = gp
        self._state_space_dim = sum(self.gp.state_space_dim())
        self.keep_dims = np.array(keep_dims)
        self.whiten_space = whiten_space

    @property
    def temporal_output_dim(self):
        """ Returns the full state.  """
        return self.keep_dims.shape[0]

    @property
    def _output_dim(self):
        return self.temporal_output_dim * self.spatial_output_dim

    def H(self, x, X_s, t):

        # Observe both f and df
        H_t = np.eye(self._state_space_dim)
        # only keep the deriatives that we care about
        H_t = H_t[self.keep_dims]

        #When there are no spatial points there is no need to permute
        #as it will automatically be in time-latent format
        if X_s is None:
            return H_t

        # need to permute from latent-space-state to latent-state-space
        # TODO: assuming that latent = 1 and we are treating state as latent
        Ns = self.spatial_output_dim * X_s.shape[0]
        P = self.temporal_output_dim
        full_P = self._state_space_dim

        H = data_order_to_output_order(full_P, Ns).T

        idx = np.tile(np.arange(Ns), P) + np.repeat(self.keep_dims, Ns)*Ns
        H = H[idx]


        return H

class EulerMaruyama(SDE):
    def __init__(self, base_sde):
        self.base_sde = base_sde

    def H(self, x, X_s, t):
        return self.base_sde.H(x, X_s, t)

    def P_inf(self, x, X_s, t):
        return self.base_sde.P_inf(x, X_s, t)

    def m_inf(self, x, X_s, t):
        return self.base_sde.m_inf(x, X_s, t)

    def f_dt(self, x, X_s, t, dt):
        base_f = self.base_sde.f(x, X_s, t)
        return x + base_f * dt

    def Sigma_dt(self, x, X_s, t, dt):
        L = self.base_sde.L(x, X_s, t)
        Q = self.base_sde.Q(x, X_s, t)

        return L @ Q @ L.T * dt

    def _f(self, x, t):
        return self.base_sde._f(x, t)

