import jax
import jax.numpy as np
import chex

from . import Transform, LinearTransform
from . import Independent
from .. import Parameter

from ..dispatch import evoke


class LatentVariable(LinearTransform):
    def __init__(self, base_gp, latent_variable, with_hessian=False):
        self._parent = Independent([base_gp, latent_variable])

        # flag whether or not hessian should be computed in approximation
        self.with_hessian = with_hessian

class ConcatenateLatentVariable(LatentVariable):
    """ Model X = [X, W]. """
    def transform_x(self, X, W):
        return np.hstack([X, W])

class AdditiveLatentVariable(LatentVariable):
    """ Model X = X+W. Assumes W is same dim as X """
    def transform_x(self, X, W):
        chex.assert_equal_shape([X, W])

        return X + W

class UncertainInput(LinearTransform):
    def __init__(self, base_gp, variance = None):
        self._parent = Independent([base_gp])

        if variance is None:
            variance = 1.0

        self.var_param = Parameter(
            np.array(variance), 
            constraint='positive', 
            name ='UncertainInput/variance', 
            train=True
        )

    @property
    def full_transform(self):
        return True

    def transform_diagonal(self, mu, var, data):
        return self.transform(mu, var, data)

    def transform(self, mu, var, data):
        return jax.vmap(
            self.transform_single, 
            [0, 0, 0]
        )(
            mu, 
            var,
            self.var_param.value
        )

    def transform_single(self, mu, var, input_var):
        chex.assert_rank([mu, var], [2, 3])
        f =  mu[0]
        df = mu[1]

        var_f = var[0][0][0]
        var_df = var[0][1][1]


        trans_mu = f
        #trans_var = var[0][0][0] + input_var * (df**2 + var_df)
        trans_var = var[0][0][0] + input_var * (df**2)

        trans_mu = np.reshape(trans_mu, [1, 1])
        trans_var = np.reshape(trans_var, [1, 1, 1])

        chex.assert_rank([trans_mu, trans_var], [2, 3])
        return trans_mu, trans_var


