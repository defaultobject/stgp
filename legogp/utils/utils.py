import jax
import jax.numpy as np
import objax
from .. import Parameter
from batchjax import BatchType
import numpy as onp

""" General Utils. """
def ensure_module_list(arr: list) -> objax.ModuleList:
    if arr is None: return arr

    if type(arr) is not objax.ModuleList:
        if type(arr) is not list:
            arr = [arr]

        arr = objax.ModuleList(arr)

    return arr


def ensure_array(a):
    return np.array(a)

def ensure_float(a):
    return float(a)

def key_that_ends_with(d: dict, k: str):
    for key in d.keys():
        if key.endswith(k):
            return key
    return None


def can_batch(module_list):
    # if all types are the same then batch
    first_type = type(module_list[0])

    if all(type(m) == first_type for m in module_list):
        return True

    return False

def get_batch_type(module_list):
    if can_batch(module_list):
        return BatchType.OBJAX

    return BatchType.LOOP


def match_suffix(s, arr, return_single = True):
    res =  [a for a in arr if a.endswith(s)]

    if return_single:
        assert len(res) == 1
        return res[0]

    return res

def vc_keep_vars(vc, keys):
    vc_new = objax.VarCollection()

    for k in keys:
        vc_new.update((name, v) for name, v in vc.items() if name in keys)

    return vc_new

def vc_remove_vars(vc, keys):
    vc_new = objax.VarCollection()

    for k in keys:
        vc_new.update((name, v) for name, v in vc.items() if name not in keys)

    return vc_new

def _summarize_var(v):
    if onp.sum(v.shape) > 5:
        return v.shape

    elif isinstance(v, objax.BaseVar):
        return onp.array(v.value)

    return onp.array(v)

def get_parameters(m, scope=''):
    parameters = {}

    #imitate objax scoping so that parameters are consistently printed
    scope += f'({m.__class__.__name__}).'

    for k, v in m.__dict__.items():
        if isinstance(v, objax.BaseVar):
            #ignore statevars as they are not trained
            if not isinstance(v, objax.StateVar):
                parameters[scope + k] = _summarize_var(v)

        elif isinstance(v, Parameter):
            if v.name == None:
                parameters[scope + k] = _summarize_var(v.value)
            else:
                parameters[v.name] = _summarize_var(v.value)

        elif isinstance(v, objax.ModuleList):
            for p, v in enumerate(v):
                parameters.update(
                    get_parameters(v, scope=f'{scope}[{p}]')
                )

        elif isinstance(v, objax.Module):
            if k == '__wrapped__':
                parameters.update(
                    get_parameters(v, scope=scope[:-1])
                )
            else:
                parameters.update(
                    get_parameters(v, scope=scope + k)
                )

    return parameters
