import numpy as np

_REGISTERED = {}

class _REGISTERED_KEY:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

class _DISPATCHER:
    @staticmethod
    def match(key:_REGISTERED_KEY, *args, **kwargs):
        for x, y in zip(key.args, args):
            if x != y:
                return False

        for k, i in key.kwargs.items():
            if k not in kwargs.keys():
                return False

            if kwargs[k] != key.kwargs[k]:
                return False

        return True


def obj_dispatch(*args, **kwargs):
    def decorator(obj):
        k = _REGISTERED_KEY(args, kwargs)
        _REGISTERED[k] = obj

        return obj

    return decorator

def obj_find(*args, **kwargs):
    for k, item in _REGISTERED.items():
        if _DISPATCHER.match(k, *args, **kwargs):
            return item

    raise RuntimeError()
