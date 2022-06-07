import objax
from batchjax import batch_or_loop, BatchType

from typing import Union, Optional, List
from .utils import can_batch, get_batch_type
from ..dispatch import evoke


def get_modules_at_i(module_arr, i):
    arr = []
    for m_arr in module_arr:
        arr.append(m_arr[i])
    return arr

def batch_over_module_types(        
    evoke_name: str,
    evoke_params: list,
    module_arr: Union[List[objax.ModuleList], objax.ModuleList],
    fn_params: list,
    fn_axes: list,
    dim: int,
    out_dim :int
):
    """ 
    Helper function to batch over a module.

    Assumes that the evoke function takes
        <evoke_name>, *evoke_params, module
    """

    if type(module_arr) != list:
        module_arr = [module_arr]

    # make sure lists are of same dimension
    list_dim = len(module_arr[0])
    for m in module_arr:
        assert len(m) == list_dim

    num_types = len(module_arr)

    # if all modules are the same we only need the first object
    #   and then we can batch it
    # otherwises we need the whole array and we will loop through them all
    if can_batch(module_arr):
        pred_fn = evoke(evoke_name, *evoke_params, *get_modules_at_i(module_arr, 0))
        pred_axes = None
    else:
        pred_fn = [
            evoke(evoke_name, *evoke_params, *get_modules_at_i(module_arr, i)) for i in range(list_dim)
        ]
        pred_axes = 0

    fn = lambda pred_fn, *args: pred_fn(*args)

    # Compute prediction for each likelihood-prior pair
    return batch_or_loop(
        fn,
        [pred_fn, *fn_params],
        [pred_axes, *fn_axes],
        dim=dim,
        out_dim=out_dim,
        batch_type = get_batch_type(module_arr)
    )
