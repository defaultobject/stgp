import jax
import jax.numpy as np
import chex

from . import Transform, LinearTransform
from . import Independent
from .. import Parameter

from ..dispatch import evoke


class LatentVariable(LinearTransform):
    def __init__(self, base_gp, latent_variable, deep_kernel):
        self._parent = Independent([base_gp, latent_variable])
        self.deep_kernel = deep_kernel

class ConcatenateLatentVariable(LatentVariable):
    pass


class UncertainInput(LinearTransform):
    def __init__(self, base_gp, variance = None):
        self._parent = Independent([base_gp])

        if variance == None:
            variance = 1.0

        self.var_param = Parameter(
            np.array(variance), 
            constraint='positive', 
            name ='UncertainInput/variance', 
            train=True
        )

    def transform_diagonal(self, mu, var):
        return self.transform(mu, var)

    def transform(self, mu, var):
        chex.assert_rank([mu, var], [2, 3])
        f =  mu[0]
        df = mu[1]

        var_f = var[0][0][0]
        var_df = var[0][1][1]

        input_var = self.var_param.value

        trans_mu = f
        #trans_var = var[0][0][0] + input_var * var_df * df**2
        trans_var = var[0][0][0] + input_var * (df**2 + var_df)
        #trans_var = var[0][0][0] + input_var * df*var_f*df

        trans_mu = np.reshape(trans_mu, [1, 1])
        trans_var = np.reshape(trans_var, [1, 1, 1])

        chex.assert_rank([trans_mu, trans_var], [2, 3])
        return trans_mu, trans_var


