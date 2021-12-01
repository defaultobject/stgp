from ..obj_dispatch import obj_dispatch, obj_find
from ..inference import Batch
from ..core import GPPrior

def GP(*args, inference=Batch(), **kwargs):
    # Y is passed either explictly through kwargs to implitly through args
    if 'Y' in kwargs.keys() or len(args) > 1:
        return obj_find('Model', inference)(*args, **kwargs)
    else:
        return GPPrior(*args, **kwargs)

