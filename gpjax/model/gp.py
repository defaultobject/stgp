import objax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple

from ..decorators import strict_mode_check, ensure_data
from ..obj_dispatch import obj_dispatch, obj_find

from ..kernel import Kernel
from ..inference import Batch
from ..transform import Transform
from ..node import Node


class Model(objax.Module):
    def __init__(self):
        super(Model, self).__init__()

def GP(*args, inference=Batch(), **kwargs):
    if len(args) > 0:
        if isinstance(args[0], Model) or isinstance(args[0], Transform):
            return NodeGP(*args, **kwargs)

    if 'parent' in kwargs:
        if isinstance(kwargs['parent'], Model):
            return NodeGP(*args, **kwargs)


    return obj_find('Model', inference)(*args, **kwargs)

class NodeGP(Model):
    def __init__(self, parent: Model, X:Optional[np.ndarray]=None, Y:Optional[np.ndarray]=None, likelihood: 'Likelihood'=None, kernel: 'Kernel'=None, whiten=False):
        self.parent = parent


