""" Helper functions and classes for classifying if a model is linear or non linear """
from .models import Model
from ..transforms import LinearTransform, NonLinearTransform
from .gp_prior import GPPrior

class LinearModel(Model):
    def __init__(self, prior):
        self._parent = prior

class NonLinearModel(Model):
    def __init__(self, prior):
        self._parent = prior

def get_model_type(prior):
    cur_prior = prior
    is_linear = True
    while True:
        if isinstance(cur_prior, GPPrior):
            break

        if isinstance(cur_prior, LinearTransform):
            cur_prior = prior.parent
        else:
            is_linear = False
            break
    
    if is_linear:
        return LinearModel(prior)

    return NonLinearModel(prior)

