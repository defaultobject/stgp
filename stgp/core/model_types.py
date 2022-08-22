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

def _get_linear_model_part(prior):

    parent_prior = prior.parent

    if isinstance(prior, MultiOutput):
        # when a multi output we simple get the linear part for each
        #   output separetely
        return [
            _get_linear_model_part(m) for m in parent_prior
        ]
        breakpoint()

    elif prior.is_base:
        return prior

    elif isinstance(prior, LinearTransform):
        if isinstance(get_model_type(prior),LinearModel):
            return prior

    return _get_linear_model_part(parent_prior)

def _get_non_linear_model_part(prior) -> list:
    parent_prior = prior.parent
    parent_model_type = get_model_type(parent_prior)
    parent_is_linear = isinstance(get_model_type(parent_prior), LinearModel)

    # checks for base of the recursion
    if isinstance(prior, LinearTransform) and parent_is_linear:
        return []
    elif prior.is_base:
        raise RuntimeError()


    # we need to check if multioutput before as the parent of a multioutput
    #   is a module list
    if isinstance(prior, MultiOutput):
        # when a multi output we simple get the non linear part for each
        #   output separetely
        return [[
            _get_non_linear_model_part(m) for m in parent_prior
        ], prior]
            
    elif parent_is_linear:
        return [prior]

    # only here is parent is not linear 
    parent_non_linear_part = _get_non_linear_model_part(parent_prior)
    parent_non_linear_part.append(prior)
    return parent_non_linear_part


def get_linear_model_part(prior):
    """
    A prior consists of a set of transformations like:
        T_2(T_1(GP))
    The linear part is the prior transformed after (potentially zero, in which case returns just the prior) a set of linear transforms.
    """
    linear_part = _get_linear_model_part(prior)

    return linear_part


def get_non_linear_model_part(prior) -> list:
    """
    A prior consists of a set of transformations like:
        T_2(T_1(GP))
    The non_linear part is the prior that transforms a linear part
    """

    non_linear_part = _get_non_linear_model_part(prior)

    return non_linear_part
