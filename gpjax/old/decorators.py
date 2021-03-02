from .parameter import Parameter
from .data import Data, PlaceholderData
from .module import Registered

import jax
import jax.numpy as jnp
from jax import value_and_grad

import numpy as onp


import inspect
import functools

USE_CACHE = True

FUNC_CACHE = {}


def clear():
    global FUNC_CACHE
    global JIT_CACHE
    FUNC_CACHE = {}
    JIT_CACHE = {}


def jit_with_scope(*jit_args, **jit_kwargs):
    """
    Checks if the first argument is of type Data, else creates a Placeholder data argument
    """

    if len(jit_args) != 0:
        static_argnums = []
    else:
        # +1 because the jitted function has params ad first argument
        if "static_argnums" in jit_kwargs.keys():
            static_argnums = onp.array(jit_kwargs["static_argnums"]) + 1
        else:
            static_argnums = []

    def inner(func):
        argspec = inspect.getfullargspec(func)

        def update(params, *args):
            Parameter.PARAM_DICT.update(params)

            for arg in args:
                if isinstance(arg, Registered):
                    arg.update(params)

        def fn_to_jit(params, *args, **kwargs):
            update(params, *args)
            # update(params, kwargs.items())

            return func(*args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cached = Parameter.PARAM_DICT.copy()

            if USE_CACHE and (func in FUNC_CACHE):
                jit_fn = FUNC_CACHE[func]
            else:
                jit_fn = jax.jit(fn_to_jit, static_argnums=static_argnums)
                FUNC_CACHE[func] = jit_fn

            # print(Parameter.PARAM_DICT)
            val = jit_fn(Parameter.PARAM_DICT, *args, **kwargs)

            update(cached, *args)
            # update(cached, kwargs.items())
            return val

        return wrapper

    if len(jit_args) != 0:
        return inner(*jit_args, **jit_kwargs)

    return inner


JIT_CACHE = {}


def grad_with_scope(func):
    """
    Checks if the first argument is of type Data, else creates a Placeholder data argument
    """
    argspec = inspect.getfullargspec(func)

    def update(params, param_dict, *args):

        param_dict.update(params)
        Parameter.PARAM_DICT.update(param_dict)

        for arg in args:
            if isinstance(arg, Registered):
                arg.update(param_dict)

    def inner(*args, params=None, **kwargs):
        if params is None:
            params = Parameter.PARAM_DICT

        # We have to pass through param_dict so fn_to_diff is functional and does no depend on the global state
        def fn_to_diff(params, param_dict, *args, **kwargs):
            update(params, param_dict, *args)
            # update(params, **kwargs)

            return func(*args, **kwargs)

        keys_hash = str(list(params.keys()))
        type_hash = str([type(v) for i, v in params.items()])
        keys_hash = keys_hash + type_hash
        if USE_CACHE and (func in JIT_CACHE) and (keys_hash in JIT_CACHE[func]):
            jit_fn = JIT_CACHE[func][keys_hash]
        else:
            jit_fn = jax.jit(jax.value_and_grad(fn_to_diff))

            if func not in JIT_CACHE:
                JIT_CACHE[func] = {}

            JIT_CACHE[func][keys_hash] = jit_fn

        cached = Parameter.PARAM_DICT.copy()
        update(cached, Parameter.PARAM_DICT, *args)
        # cached = params.copy()

        val, grad = jit_fn(params, Parameter.PARAM_DICT, *args)

        update(cached, Parameter.PARAM_DICT, *args)
        return val, grad

    return inner


def return_gradients(
    func,
):
    def inner(*args, params=None, return_grad=True, jit=True, **kwargs):
        if return_grad:
            return grad_with_scope(func)(*args, params=params, **kwargs)

        elif jit:
            return jit_with_scope(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return inner


def set_defaults_from_self(func):
    """
    Replace any unpassed variables with their value in self
    """
    argspec = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # get arguments that have not been passed a value
        unpassed_positional_args = argspec.args[len(args) :]

        # args[0] is the object (`self`), so get the unpassed value from self
        new_args = []
        for a in unpassed_positional_args:
            if a not in kwargs:
                if hasattr(args[0], a):
                    new_args.append((a, getattr(args[0], a)))
                else:
                    # use the default
                    pass

        kwargs.update(new_args)

        # call function with defaults from self
        return func(*args, **kwargs)

    return wrapper


def ensure_data_passed(func):
    """
    Checks if the first argument is of type Data, else creates a Placeholder data argument
    Only for X, assumes that Y is None
    """
    argspec = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)

        if not issubclass(type(args[1]), Data):

            data = PlaceholderData(args[1], None)

            args[1] = data

        return func(*args, **kwargs)

    return wrapper
