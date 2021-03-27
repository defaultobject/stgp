from .. import settings
from . import Quadrature
from . import MonteCarlo

def get_approximator():
    if settings.integral_approximator == 'quadrature':
        return Quadrature()
    elif settings.integral_approximator == 'monte_carlo':
        return MonteCarlo()

    raise RuntimeError()

