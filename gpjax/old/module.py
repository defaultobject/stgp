from .parameter import Parameter
import jax
from jax import jit, partial
import jax.numpy as np
import jax.tree_util as jtree
from jax.tree_util import register_pytree_node
from jax.tree_util import tree_flatten, tree_unflatten


import typing
from typing import List, Callable

import abc
from abc import ABC
from abc import abstractmethod

import uuid


import inspect


class Registered:
    def __init__(self):
        self.register(self.__class__)

    def save_inputs_to_properties(self, inputs):
        argspec = inspect.getfullargspec(self.__class__.__init__)
        ignore_keys = ["self"]

        for n in argspec.args:
            # skip self
            if n in ignore_keys:
                continue
            if n not in inputs.keys():
                continue
            setattr(self, n, inputs[n])

    def register(self, class_name: str) -> None:
        if class_name not in jtree._registry:

            argspec = inspect.getfullargspec(class_name.__init__)

            ignore_keys = ["self", "name", "trainable", "params"]

            debug = False

            def module_flatten(module):
                """ Specifies how to flatten a distribution. """

                if debug:
                    print("=============================")
                    print("flatten: ", class_name)
                    print("flatten: ", module)

                arg_names = [n for n in argspec.args if (n not in ignore_keys)]

                params = {}

                param_keys = list(params.keys())

                aux_data = {}
                aux_data["__input__"] = {}
                aux_data["__additional__"] = {}

                # go through every argument of the __init__ method of module
                #   if arg is a jax type add to params
                #   if arg is an array of jax types add to params
                #   else asdd to the auxillary array
                for arg in arg_names:
                    obj = getattr(module, arg)
                    # jax supports dict, array and objects
                    if isinstance(obj, Registered):
                        if debug:
                            print("Registered: ", arg, obj)

                        if type(obj) not in jtree._registry:
                            raise RuntimeError(type(obj))

                            # print(tree_flatten(obj))
                        params[arg] = obj
                        # params[arg] = obj
                    elif isinstance(obj, Parameter):
                        if debug:
                            print("Parameter: ", arg, obj.val)
                            print("Parameter shape: ", arg, obj.raw_value.shape)
                            print("Parameter name: ", arg, obj.name)
                        params[arg] = obj.raw_value
                        aux_data["__additional__"][arg] = obj.name
                    elif (type(obj) is list) and np.all(
                        [isinstance(a, Registered) for a in obj]
                    ):
                        if debug:
                            print("List: ", arg)
                        params[arg] = obj
                    else:
                        if debug:
                            print("aux: ", arg)
                        aux_data["__input__"][arg] = obj

                aux_data["__keys__"] = list(params.keys())
                aux_data["__param_keys__"] = param_keys

                if debug:
                    print("__keys__: ", aux_data["__keys__"])
                    print("__param_keys__: ", aux_data["__param_keys__"])
                    print("__input__: ", aux_data["__input__"].keys())
                    # print(params)

                return (params.values(), aux_data)

            def module_unflatten(aux_data, *args):
                """ Specifies how to unpack into a class. """
                if debug:
                    print("=============================")
                    print("unflatten: ", class_name)
                    print("__keys__", aux_data["__keys__"])
                    print("param_keys", aux_data["__param_keys__"])

                keys = aux_data["__keys__"]
                param_keys = aux_data["__param_keys__"]
                jit_params = dict(zip(keys, args[0]))

                params = {k: jit_params[k] for k in param_keys}
                jax_objects_inputs = {
                    k: jit_params[k] for k in keys if (k not in param_keys)
                }

                if False and debug:
                    print(jit_params)
                    print(jax_objects_inputs)

                static_inputs = aux_data["__input__"]
                name = str(uuid.uuid4())

                inputs = jax_objects_inputs
                inputs.update(static_inputs)

                if "params" in argspec.args:
                    inputs.update({"params": params})
                else:
                    inputs.update(params)

                if "trainable" in argspec.args:
                    # we do not want to create new parameters
                    # fix parameter names

                    new_class = class_name(**inputs, trainable=False)
                    for p in keys:
                        try:
                            if debug:
                                print(
                                    new_class,
                                    " ",
                                    p,
                                    " ",
                                    aux_data["__additional__"][p],
                                )
                            getattr(new_class, p).name = aux_data["__additional__"][p]
                        except KeyError as e:
                            print(new_class)
                            print(p)
                            print(inputs)
                            print(aux_data["__additional__"])
                            raise e

                    return new_class
                else:
                    return class_name(**inputs, name=name)

            self._module_flatten = module_flatten
            self._module_unflatten = module_unflatten

            # Global registration
            register_pytree_node(
                class_name,
                module_flatten,  # tell JAX what are the children nodes
                module_unflatten,  # tell JAX how to pack back into a class_name
            )


class Module(Registered):
    """
    Any child class will be auto-registered as a jax object, allowing them to be passed into @jit'ed functions.

    The children class must only accept two arguments into init:
        params (dict) -> a dictionary of parameters and values
        meta (dict) -> a dictionary of arguments that are non-jax parameters
        name (str) -> a unique name for the object (if None a unique ID will be created)

    And it must overwrite get_default_params returing a default value for all required parameters inside params

    SHARP BITS:
        any parameters that are passed in through params, should not be changed before being registered
    """

    def __init__(self, name: str) -> None:
        super(Module, self).__init__()
        # keep track of Module's params
        self.PARAMS = {}
        self.name = name

    def parameter(
        self,
        val: np.ndarray,
        meta=None,
        constraint: Callable = None,
        train: bool = True,
        param_name: str = None,
        module_name: str = None,
        scope: str = None,
    ) -> Parameter:
        unique_name = module_name + "/" + param_name

        new_param = Parameter(
            init=val,
            meta=meta,
            shape=None,
            constraint=constraint,
            transform=None,
            name=unique_name,
            scope=scope,
            train=train,
        )

        self.PARAMS[param_name] = new_param

        return self.PARAMS[param_name]

    def update(self, params):
        # pass
        # update own paramets
        DEBUG_FLAG = False
        for name, val in self.PARAMS.items():
            if DEBUG_FLAG:
                if val.name in Parameter.PARAM_DICT.keys():
                    if np.sum(params[val.name].shape) > 5:
                        print("---- ", name, np.sum(params[val.name]))
                    else:
                        print("---- ", name, params[val.name])
                else:
                    print("---- ", name)
            if val.name in Parameter.PARAM_DICT.keys():
                val.raw = params[val.name]
            else:
                if DEBUG_FLAG:
                    print("Not in param_dict: ", val.name)

        # update any registered objects

        for a in vars(self):
            if a.startswith("__"):
                continue

            _attr = getattr(self, a)
            # print(' - ', a)

            if isinstance(_attr, list):
                if isinstance(_attr[0], Registered):
                    for l in _attr:
                        l.update(params)

            if isinstance(_attr, Registered):
                if DEBUG_FLAG:
                    print("Updating: ", _attr)
                _attr.update(params)

    def get_params(self):
        """
        Returns the raw (un-constrained) parameters of Module
        """
        d = {}

        for name, param in self.PARAMS.items():
            d[name] = param.raw_value

        return d

    @abstractmethod
    def get_default_params(self):
        return {}

    def get_param_name_mappings(self):
        # return self.get_default_params().keys()
        names = {}
        for name, param in self.PARAMS.items():
            n = param.name
            names[name] = n
        return names
