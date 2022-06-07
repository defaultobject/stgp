from . import Transform, Independent

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

