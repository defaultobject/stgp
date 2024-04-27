from functools import partial
from jax import jit, grad
import jax.numpy as np
import numpy as onp
import os

from interpax import interp1d


@partial(jit, static_argnums=(0, 2))
def custom_bessel_ive(order, x, interp):
    """ Approximate modified Bessel function of the first kind """

    BESSEL_X = np.array(onp.load('/Users/oliverhamelijnck/Documents/projects/stgp/stgp/computation/custom/precomputed_bessel_x.npy'))

    BESSEL_Y = np.array(onp.load('/Users/oliverhamelijnck/Documents/projects/stgp/stgp/computation/custom/precomputed_bessel_y.npy'))

    if interp == 0:
        return np.interp(x, BESSEL_X, BESSEL_Y[order])

    return interp1d(x, BESSEL_X, BESSEL_Y[order], method="cubic")
    
    

