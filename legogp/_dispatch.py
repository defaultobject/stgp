from multipledispatch import dispatch as dispatcher
from functools import partial
import inspect

#method multiple dispatch
legogp_namespace = dict()

# dispatch using the gpjax namespace

#dispatcher = partial(dispatcher, namespace=legogp_namespace)

dispatch= partial(dispatcher, namespace=legogp_namespace)

def _dispatch(*types, **kwargs):

    types = [t for t in types]
    return dispatcher(*types, **kwargs)




def evoke(fn_name):
    """Get a dispatched function."""

    if fn_name in legogp_namespace:
        return legogp_namespace[fn_name]
    else:
        raise RuntimeError()



