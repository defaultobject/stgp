import types
import inspect

_REGISTERED = {}

def _ensure_str(k):
    if type(k) is not str:
        #the passed k is either a class or an class instance / object

        if not isinstance(k, type):
            #the passed k in an object
            k = type(k)

        return k.__name__
    return k

def _try_match(x, y):
    # check if y is a child class of x

    if inspect.isclass(x):
        _x = x
    else:
        _x = type(x)

    if inspect.isclass(y):
        _y = y  
    else:
        _y = type(y)

    # avoid catch alls
    if _x != object and _y != object:
        if _x != _y:
            # only check for inherentance
            # as string comparision will catch same types

            if issubclass(_y, _x):
                return True

    if _ensure_str(x) != _ensure_str(y):
        return False

    return True


class _REGISTERED_KEY:
    def __init__(self, obj, args, kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

class _DISPATCHER:
    @staticmethod
    def match(key:_REGISTERED_KEY, *args, **kwargs):
        for x, y in zip(key.args, args):
            if not _try_match(x, y):
                return False

        for k, i in key.kwargs.items():
            if _ensure_str(k) not in kwargs.keys():
                return False

            if not _try_match(kwargs[k], key.kwargs[k]):
                return False

        return True

def dispatch(*args, **kwargs):
    def decorator(obj):
        if isinstance(obj, types.FunctionType):
            # dispatch a function

            # add fn name to the key
            k = _REGISTERED_KEY(obj, [obj.__name__, *args], kwargs)

        elif inspect.isclass(obj):
            # dispatch a class
            # Although the class is passed it will not be used to construct the key
            k = _REGISTERED_KEY(obj, args, kwargs)

        else:
            raise RuntimeError(f'Cannot Dispatch {obj}')

        _REGISTERED[k] = obj

        return obj

    return decorator

def evoke(*args, **kwargs):
    matched_items = []
    for k, item in _REGISTERED.items():
        if _DISPATCHER.match(k, *args, **kwargs):
            matched_items.append(item)

    if len(matched_items) == 1:
        return matched_items[0]

    if len(matched_items) >= 1:
        breakpoint()

    raise RuntimeError(f'Cannot evoke {args}, {kwargs}')
