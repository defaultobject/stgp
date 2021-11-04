from multipledispatch import dispatch as dispatcher
from functools import partial
import inspect

#method multiple dispatch
gpjax_namespace = dict()

# dispatch using the gpjax namespace

#dispatcher = partial(dispatcher, namespace=gpjax_namespace)

dispatch= partial(dispatcher, namespace=gpjax_namespace)

def _dispatch(*types, **kwargs):

    types = [t for t in types]
    return dispatcher(*types, **kwargs)




def evoke(fn_name):
    """Get a dispatched function."""

    if fn_name in gpjax_namespace:
        return gpjax_namespace[fn_name]
    else:
        raise RuntimeError()



