import jax
import jax.numpy as np
import objax

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
    return False


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
