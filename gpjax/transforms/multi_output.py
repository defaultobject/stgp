"""Multi-output/task specific transforms."""
from .transform import Transform


class LMC(Transform):
    r"""
    Linear model of coregionilisation.

    Generative model:
        f_p = \sum w_{p,q} g_q
    """


class GPRN(Transform):
    r"""
    Gaussian Process Regression Network.

    Generative model:
        f_p = \sum w_{p,q}(x) g_q(x)
    """


class ConstrainedLMC(Transform):
    r"""
    Constrained Linear model of coregionilisation.

    Generative model:
        f_p = \sum w_{p,q} g_q
    """


class ConstrainedGPRN(Transform):
    r"""
    Constrained Gaussian Process Regression Network.

    Generative model:
        f_p = \sum w_{p,q}(x) g_q(x)
    """
