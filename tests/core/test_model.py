import jax.numpy as np
import objax

from stgp import Parameter
from stgp.core import Model


def test_model_fix_and_release_restore_parameter_trainability():
    class Child(objax.Module):
        def __init__(self):
            self.trainable = Parameter(np.array(1.0))
            self.fixed = Parameter(np.array(2.0), train=False)

    class TestModel(Model):
        def __init__(self):
            self.children = objax.ModuleList([Child()])

    model = TestModel()
    child = model.children[0]

    model.fix()
    model.fix()
    assert not child.trainable.is_trainable
    assert not child.fixed.is_trainable

    model.release()
    assert child.trainable.is_trainable
    assert not child.fixed.is_trainable

    model.release()
    assert child.trainable.is_trainable
    assert not child.fixed.is_trainable
