from . import Sparsity

import typing
from typing import Optional


class NoSparsity(Sparsity):
    def __init__(self, name: Optional[str] = "sparsity"):
        super(Sparsity, self).__init__(name)
