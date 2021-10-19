import objax
import jax
import jax.numpy as np
import typing
from typing import Callable


def get_obj_at_idx(obj_arr: list, axis_arr: list, i: int) -> list:
    """ Get the objects for the batch index i. """
    indexed_objects = []
    num_obj = len(obj_arr)

    for j in range(num_obj):
        # Get the batching axis for object j
        axis_j = axis_arr[j]


        if  axis_j == None:
            # When None this object is not batched
            indexed_objects.append(
                obj_arr[j]
            )
        elif axis_j == 0 :
            # Index the first axis
            # special case so that we can support passing both lists and numpy arrays
            indexed_objects.append(
                obj_arr[j][i]
            )

        elif axis_j == 1:
            indexed_objects.append(
                obj_arr[j][:, j]
            )
        else:
            raise NotImplementedError()
            # Index an arbitrary axis
            # Must be an array
            A = np.hstack([obj_arr[j]*(k+1) for k in range(5)])
            # Implements indexing like a[:, :, 0, :, :] for an arbitrary number of axes
            breakpoint()
            indexed_objects.append(
                obj_arr[j][[slice(None)] * (obj_arr[j].ndim - 1) + [i]]
            )
    return indexed_objects


def loop_or_batch(fn: Callable, obj_arr: list, axis_arr: list, loop_len: int) -> Callable:
    """
    Automatically batch call fn.
    The syntax follows that of jax.vmap:
    Args:
        fn: Callable = the function to batch
        obj_arr: list = A list of objects, where each object is to be distributed (as described by axis_arr)
        axis_arr: list = a list of indexes of which axis to bach over
        loop_len: the number of elements to batch over
    """
    if False :
        res_arr =[]
        for i in range(loop_len):
            obj_at_i = get_obj_at_idx(
                obj_arr,
                axis_arr,
                i
            )
            res_i = fn(*obj_at_i)
            res_arr.append(res_i)
    else:
        
        num_objects = len(obj_arr)

        obj_to_batch_flag = [type(obj_i) == objax.ModuleList  for obj_i in obj_arr]

        obj_to_converted_batched = [Batched(obj_arr[i]) if obj_to_batch_flag[i] else None for i in range(num_objects) ]

        #Manual with statement enter
        batched_obj_enter = [obj_to_converted_batched[i].__enter__()  if obj_to_batch_flag[i] else None for i in range(num_objects)]

        vmap_obj_arr = []
        vmap_axis_arr = []

        for i in range(num_objects):
            if obj_to_batch_flag[i]:
                vmap_obj_arr.append(batched_obj_enter[i])
                vmap_obj_arr.append(batched_obj_enter[i].get_vars())

                vmap_axis_arr.append(None)
                vmap_axis_arr.append(axis_arr[i])
            else:
                vmap_obj_arr.append(obj_arr[i])
                vmap_axis_arr.append(axis_arr[i])

        # append so that batched_fn is functional
        vmap_obj_arr.append(obj_to_batch_flag)
        vmap_axis_arr.append(None)


        def batched_fn(*args):
            obj_to_batch_flag = args[-1]
            num_objects = len(obj_to_batch_flag)

            args_with_vars = []
            j = 0
            for i in range(num_objects):
                if obj_to_batch_flag[i]:
                    obj = args[j]
                    obj_vars = args[j+1]
                    obj.set_vars(obj_vars)
                    obj = obj.get_obj()
                    args_with_vars.append(obj)

                    j+=2
                else:
                    args_with_vars.append(args[j])
                    j += 1

            return fn(*args_with_vars)

        res_arr = jax.vmap(batched_fn, vmap_axis_arr)(*vmap_obj_arr)

        # Manual with statement exit
        batched_obj_exit = [obj_to_converted_batched[i].__exit__()  if obj_to_batch_flag[i] else None for i in range(num_objects)]


    return res_arr

class batch():
    def __init__(self, func):
        self.func = func
        self.variable = self.func.__name__
        self.original_getter = None


    def getter(self, obj):
        return getattr(obj, f'raw_{self.variable}').value

    def __get__(self, obj, objtype=None):
        return self.func(obj, lambda: self.getter(obj))

    def __set__(self, obj, val):
        if val is None:
            self.getter = self.original_getter

        else:
            self.original_getter = self.getter
            self.getter = lambda obj: val


class Batched():
    def __init__(self, obj, num_probe_vectors=1):
        self.obj = obj
        self.batched_obj = Batcher(obj)
        self.batched_vars = None

    def get(self, name):
        """Get jax variable"""
        key = utils.key_that_ends_with(self.batched_vars, name)
        return self.batched_vars[key] 

    def get_vars(self):
        return self.batched_obj.get_batched_vars()

    def get_obj(self):
        return self.obj[0]

    def set_vars(self, var):
        obj = self.obj[0]
        for key in var.keys():
            name = key.split('raw_')[-1]
            setattr(obj, name, var[key])


    def __enter__(self):
        self.batched_vars = self.batched_obj.get_batched_vars()
        return self

    def __exit__(self, *args):
        #Only need to apply to the first obj
        obj = self.obj[0]
        for key in obj.vars().keys():
            #assume that key ends in raw_
            name = key.split('raw_')[-1]

            #disable batching
            setattr(obj, name, None)


class Batcher():
    def __init__(self, obj_list: objax.ModuleList):
        self.obj_list = obj_list

    def get_batched_vars(self):
        """For each variable in obj_list, construct a numpy array with the tracers so that they can be batched over."""

        all_vars = {}

        #collect vars
        for obj in self.obj_list:
            var_collection = obj.vars()
            for key in var_collection.keys():
                if key not in all_vars:
                    all_vars[key] = []

                all_vars[key].append(var_collection[key].value)

        #convert to jax array
        for obj in self.obj_list:
            var_collection = obj.vars()
            for key in var_collection.keys():
                all_vars[key] = np.array(all_vars[key])

            #only need to do for the first object as all object should have the same var_collection

        return all_vars

