from multipledispatch import dispatch
from functools import partial
import inspect

#method multiple dispatch
gpjax_namespace = dict()

# dispatch using the gpjax namespace
dispatch = partial(dispatch, namespace=gpjax_namespace)

def evoke(fn_name):
    """Get a dispatched function."""

    if fn_name in gpjax_namespace:
        return gpjax_namespace[fn_name]
    else:
        raise RuntimeError(f"Dispatch envoke: {fn_name} does not exist")

#object multiple dispatch


