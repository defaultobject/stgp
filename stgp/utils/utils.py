import jax
import jax.numpy as np
import numpy as onp
import objax
import chex
from .. import Parameter
from .. import settings
from ..core.block_types import Block, get_block_dim
from batchjax import BatchType
import numpy as onp
from ..dispatch import _ensure_str
from functools import partial
from jax import jit

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

def get_obj_type(obj):
    if _ensure_str(obj) in ['ProductLikelihood', 'GaussianProductLikelihood']:
        return [get_obj_type(lik) for lik in obj.likelihood_arr]

    return type(obj)

def do_obj_types_match(t1, t2):
    if type(t1) is list:
        if type(t2) is not list:
            return False
        if len(t1) != len(t2):
            return False
        return all(
            [do_obj_types_match(t1[i], t2[i]) for i in range(len(t1))]
        )

    if type(t2) is list:
        return False

    return t1 == t2

def do_obj_have_same_vars(t1, t2):
    t1_vars = t1.vars()
    t2_vars = t2.vars()

    # ensure t1 and t2 have the same keys
    if len(set(t1_vars.keys())-set(t2_vars.keys())) == 0:
        # check they have same shape as well
        return all([t1_vars[key].shape == t2_vars[key].shape for key in t1_vars.keys()])

    return False
    


def can_batch(module_list, debug=False):
    """
    This is an incomplete/niave way of checking if module_list consist of the same module
    """
    # if all types are the same then batch
    first_type = get_obj_type(module_list[0])

    if all(do_obj_types_match(get_obj_type(m), first_type) for m in module_list):
        if all(do_obj_have_same_vars( module_list[0], m) for m in module_list):
            return True
        else:
            return False

    return False


def get_batch_type_from_bool(b: bool):
    # settings override
    if settings.use_loop_mode:
        return BatchType.LOOP

    if b:
        return BatchType.OBJAX

    return BatchType.LOOP


def get_batch_type(module_list):
    # settings override
    if settings.use_loop_mode:
        return BatchType.LOOP

    if can_batch(module_list):
        return BatchType.OBJAX

    return BatchType.LOOP

def can_batch_np_list(X_s: list) -> bool:
    try:
        np.array(X_s)
        can_batch_xs = True
    except ValueError as e:
        can_batch_xs = False

    return can_batch_xs

def get_batch_type_over_prior_and_np_list(prior, X_s: list) -> BatchType:
    # settings override
    if settings.use_loop_mode:
        return BatchType.LOOP

    prior_can_batch = can_batch(prior)
    can_batch_xs = can_batch_np_list(X_s)

    if prior_can_batch and can_batch_xs:
        return BatchType.OBJAX
    
    return BatchType.LOOP

def fix_independent_api_and_fix_spatial_types(prior, X_s):
    """
    This is a hack that wont be needed once the 
    """
    # TODO: clean up at some point
    # this is just a way to support wrapping both SDE_GPs and Transforms of them in an Independent
    #  these should have the same api and then this would not be necessary
    if hasattr(prior.parent[0], 'kernel'):
        latent_wrapper_fn = lambda latent: latent.kernel
    else:
        latent_wrapper_fn = lambda latent: latent

    if type(X_s) is list:
        batch_over_Xs = 0
        if can_batch_np_list(X_s):
            X_s = np.array(X_s)
    else:
        batch_over_Xs = None

    return latent_wrapper_fn, X_s, batch_over_Xs
    



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

def get_parameters(m, scope='', only_fixed=False, replace_name=True, return_id=False, return_state_var=False, summarize=True):

    if summarize:
        summarize_fn = _summarize_var
    else:
        summarize_fn = lambda v: v

    parameters = {}

    #imitate objax scoping so that parameters are consistently printed
    scope += f'({m.__class__.__name__}).'

    for k, v in m.__dict__.items():
        if isinstance(v, objax.BaseVar):
            if only_fixed:
                # Only a Parameter type can be 'fixed' there skip
                continue
            #if return_state_var == False then ignore statevars as they are not trained
            if (return_state_var == True) or (not isinstance(v, objax.StateVar)):
                if return_id:
                    parameters[scope + k] = {
                        'id': id(v)
                    }
                else:
                    parameters[scope + k] = {
                        'var': summarize_fn(v),
                    }

        elif isinstance(v, Parameter):
            if only_fixed and v.is_trainable :
                continue

            if v.name == None or replace_name is False:
                if replace_name is False:
                    # A parameter object only has one objax variable (raw_var)
                    # Only_fixed is true, we are only in this if statement if v is not trainable
                    #   hence we want to return raw_var
                    # If only_fixed is False, then clamping it to only_fixed=False will make no difference
                    parameters.update(
                        get_parameters(
                            v,
                            scope=scope + k, 
                            only_fixed=False, 
                            replace_name=replace_name, 
                            return_id=return_id,
                            return_state_var=return_state_var,
                            summarize=summarize
                        )
                    )
                else:
                    if return_id:
                        parameters[scope + k] = {
                            'id': id(v)
                        }
                    else:
                        parameters[scope + k] = {
                            'var': summarize_fn(v.value),
                        }
            else:
                if return_id:
                    parameters[v.name] = {
                        'id': id(v)
                    }
                else:
                    parameters[v.name] = {
                        'var': summarize_fn(v.value),
                    }

        elif isinstance(v, objax.ModuleList):
            for p, v_i in enumerate(v):
                parameters.update(
                    get_parameters(v_i, scope=f'{scope}{k}({v.__class__.__name__})[{p}]', only_fixed=only_fixed, replace_name=replace_name, return_id=return_id, return_state_var=return_state_var, summarize=summarize)
                )

        elif isinstance(v, objax.Module):
            if k == '__wrapped__':
                parameters.update(
                    get_parameters(v, scope=scope[:-1], only_fixed=only_fixed, replace_name=replace_name, return_id=return_id, return_state_var=return_state_var, summarize=summarize)
                )
            else:
                parameters.update(
                    get_parameters(v, scope=scope + k, only_fixed=only_fixed, replace_name=replace_name, return_id=return_id, return_state_var=return_state_var, summarize=summarize)
                )

    return parameters

def get_fixed_params(m, replace_name=False):
    param_dict =  get_parameters(m, scope='', only_fixed=True, replace_name=replace_name)

    return list(param_dict.keys())


def get_var_name_with_id(model, _id, param_dict=None):
    if param_dict is None:
        param_dict = get_parameters(model, replace_name=False)

    for k, v in param_dict.items():
        if v['id'] == _id:
            return k

    raise RuntimeError(f'Did not find var with id {_id}')


def fix_prediction_shapes(mu, var, diagonal=True, squeeze=True, output_first = False):
    """
    Diagonal predictions will be in the shape
        mu: N x P x 1
        var: N x 1 x P x P or  N x P x 1 x 1
    """
    chex.assert_rank([mu, var], [3, 4])

    N = mu.shape[0]
    P = mu.shape[1]

    if mu.shape[2] != 1:
        print('fix_prediction_shapes: mu.shape[2] > 1 not supported')
        return mu, var 

    if diagonal: 
        # [N, 1, P, P] -> [N, 1, P] 
        var = np.diagonal(var, axis1=2, axis2=3)

        # remove extra dimension (fix when aggregating)
        mu = mu[..., 0] # [N, P]
        if var.shape[1] == 1:
            # will be in N x 1 x P   shape
            var = var[:, 0, ...] # [N, P]
        else:
            # will be in N x P x 1  shape
            var = var[..., 0] # [N, P]

        if output_first:
            mu = mu.T
            var = var.T
            chex.assert_shape(
                [mu, var],
                [[P, N], [P, N]],
            )
        else:
            chex.assert_shape(
                [mu, var],
                [[N, P], [N, P]],
            ) 
    else:
        mu = mu[..., 0]
        var = var[:, 0, ...]

        if output_first:
            # we can only make mu output first
            # this is needed when mu is used to compute metrics in the correct format 
            #   but we still want to log var
            mu = mu.T

            chex.assert_shape(
                [mu, var],
                [[P, N], [N, P, P]],
            )
        else:
            chex.assert_shape(
                [mu, var],
                [[N, P], [N, P, P]],
            )

    if squeeze:
        mu, var = np.squeeze(mu), np.squeeze(var) 

    return mu, var

def fix_block_shapes(m, S, data, likelihood, approximate_posterior, block_type):
    chex.assert_rank([m, S], [3, 4])
    
    # TODO: hacky hard code
    if type(block_type) == int:
        if block_type == 1:
            block_dim = 1

    else:
        block_dim = get_block_dim(
            block_type, 
            data = data, 
            likelihood = likelihood, 
            approximate_posterior = approximate_posterior
        )

    N, P, B = m.shape

    if N ==1:
        # m, S is a whole block
        if block_dim == B:
            mu, var = m, S
        elif block_dim == 1:
            # convert block to N
            mu = np.transpose(m, [2, 1, 0])
            var = np.diagonal(S[0], axis1=1, axis2=2).T[..., None, None]
        else:
            raise NotImplementedError()

    else:
        if block_dim == B:
            mu, var = m, S
        elif block_dim == P:
            mu, var = m, S
        else:
            raise NotImplementedError()

    return mu, var

_callable_hash = {}


def jit_fn_with_callable(fn, func, func_obj, args, static_args):
    """
    fn is a function that takes func as an initial argument
    """

    #jitted_fn = jit(fn, static_argnums=[0]+static_args)
    _hash = fn
    if _hash in _callable_hash.keys():
        jit_func_vars = _callable_hash[_hash] 
    else:
        def func_vars(fn, func, _func_obj, func_args, *_args):
            original_vals = _func_obj.vars().tensors()
            _func_obj.vars().assign(func_args)
            res = fn(func, *_args)
            _func_obj.vars().assign(original_vals)
            return res
        _callable_hash[_hash] = func_vars

    if True:
        res =  jax.jit(
            _callable_hash[_hash],
            static_argnums = [0, 1, 2, ]+[k+4 for k in static_args]
        )(
            fn, func, func_obj, func_obj.vars().tensors(), *args
        )
    else:
        res = _callable_hash[_hash](fn, func, func_obj, func_obj.vars().tensors(), *args)

    return res



def is_empty_space(X_space) -> bool:
    if X_space is None:
        return True

    print(X_space)

    return onp.all(onp.isnan(X_space))
