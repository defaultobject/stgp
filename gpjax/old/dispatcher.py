import numpy as np


class _RegisteredKey:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class Dispatcher:
    _REGISTERED = {}

    # cache
    _MEMOIZE_FUN = {}
    _MEMOIZE_REGISTERED_KEY = {}

    @staticmethod
    def register(key, *args, **kwargs):
        """
        matches functions grouped by key that match the types passed to *args
        """

        if key not in Dispatcher._REGISTERED:
            Dispatcher._REGISTERED[key] = {}

        def decorator(fun):
            k = _RegisteredKey(args, kwargs)
            Dispatcher._REGISTERED[key][k] = fun

            return fun

        return decorator

    @staticmethod
    def match(key: _RegisteredKey, *args, **kwargs):
        for x, y in zip(key.args, args):
            # None is a catch all
            if x is None:
                continue

            if not (x is y):
                return False

        for k, i in key.kwargs.items():
            if k not in kwargs.keys():
                return False

            if kwargs[k] != key.kwargs[k]:
                return False

        return True

    @staticmethod
    def has_catch_all(key: _RegisteredKey):
        for x in key.args:
            if x is None:
                return True
        return False

    @staticmethod
    def get_number_of_catch_all(key: _RegisteredKey):
        n = 0
        for x in key.args:
            if x is None:
                n = n + 1

        return n

    @staticmethod
    def find(key, *args, **kwargs):
        # if k has a catch all term we need to check that a more specific function cannot be matched first.
        # store a catch al untill all other functions have been checked, and if none are matched use the catch all.
        repeat = []

        for k, item in Dispatcher._REGISTERED[key].items():
            if Dispatcher.match(k, *args, **kwargs):

                if Dispatcher.has_catch_all(k):
                    repeat.append([k, item])
                else:
                    return item

        # repeat should only have one item because it is a catch all
        if len(repeat) > 1:
            # return the match with the least number of catch alls, i.e the most specific match
            num_of_catch_all = [
                Dispatcher.get_number_of_catch_all(r[0]) for r in repeat
            ]
            return repeat[np.argmin(num_of_catch_all)][1]

        if len(repeat) == 1:
            return repeat[0][1]

        return False
        # raise RuntimeError('Dispatcher key - ', key, ' with args: ',*args,' and kwargs: ', kwargs)

    @staticmethod
    def dispatch(key, *args, **kwargs):
        if key not in Dispatcher._MEMOIZE_FUN:
            Dispatcher._MEMOIZE_FUN[key] = None

        if Dispatcher._MEMOIZE_FUN[key] is not None and Dispatcher.match(
            Dispatcher._MEMOIZE_REGISTERED_KEY[key], *args, **kwargs
        ):
            return Dispatcher._MEMOIZE_FUN[key]
        else:
            fun = Dispatcher.find(key, *args, **kwargs)
            Dispatcher._MEMOIZE_FUN[key] = fun
            Dispatcher._MEMOIZE_REGISTERED_KEY[key] = _RegisteredKey(args, kwargs)
            return fun
