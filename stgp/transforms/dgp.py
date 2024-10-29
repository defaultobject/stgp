import objax
from .transform import NonLinearTransform

class DeepGP(NonLinearTransform):
    def __init__(self, chain = None):
        if chain is None:
            raise RuntimeError()
        
        self._parent = chain
    
