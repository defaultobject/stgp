from .utils import can_batch, get_batch_type
from ..dispatch import evoke
from batchjax import batch_or_loop, BatchType

def batch_over_likelihoods(        
    evoke_name: str,
    evoke_params: list,
    likelihood_arr: list,
    fn_params: list,
    fn_axes: list,
    dim: int,
    out_dim :int
):
    """ 
    Helper function to batch over likelihoods.

    Assumes that the evoke function takes
        <evoke_name>, *evoke_params, likelihood
    """

    # if all likelihooods are the same we only need the first object
    #   and then we can batch it
    # otherwises we need the whole array and we will loop through them all
    if can_batch(likelihood_arr):
        pred_fn = evoke(evoke_name, *evoke_params, likelihood_arr[0])
        pred_axes = None
    else:
        pred_fn = [evoke(evoke_name, *evoke_params, lik) for lik in likelihood_arr]
        pred_axes = 0

    fn = lambda pred_fn, *args: pred_fn(*args)

    # Compute prediction for each likelihood-prior pair
    return batch_or_loop(
        fn,
        [pred_fn, *fn_params],
        [pred_axes, *fn_axes],
        dim=dim,
        out_dim=out_dim,
        batch_type = get_batch_type(likelihood_arr)
    )
