import objax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple

from ..decorators import strict_mode_check
from ..obj_dispatch import obj_dispatch, obj_find

from ..kernel import Kernel
from ..inference import Batch


class Model(objax.Module):
    pass

def GP(*args, inference=Batch(), **kwargs):
    if len(args) > 0:
        if isinstance(args[0], Model):
            return NodeGP(*args, **kwargs)
    
    return obj_find('Model', inference)(*args, **kwargs)

class NodeGP(Model):
    def __init__(self, X=None):
        pass

@obj_dispatch(Model, 'Batch')
class BatchGP(Model):
    def __init__(self, X=None, inference=None):
        pass
