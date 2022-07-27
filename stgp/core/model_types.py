""" Helper functions and classes for classifying if a model is linear or non linear """
from .models import Model
from ..transforms import LinearTransform, NonLinearTransform, MultiOutput
from .gp_prior import GPPrior

class LinearModel(Model):
    def __init__(self, prior):
        self._parent = prior

class NonLinearModel(Model):
    def __init__(self, prior):
        self._parent = prior

def _is_prior_linear(prior):
    if prior.is_base:
        return True 

    if isinstance(prior, MultiOutput):
        # only linear if each parent is linear
        for p in prior.parent:
            if _is_prior_linear(p) == False:
                return False
        return True

    if isinstance(prior, LinearTransform):
        # is prior is linear, then one of its parents might be non-linear so we need to keep checking
        return _is_prior_linear(prior.parent)
    else:
        return False

    return False


def get_model_type(prior):
    is_linear = _is_prior_linear(prior)
    
    if is_linear:
        return LinearModel(prior)

    return NonLinearModel(prior)

