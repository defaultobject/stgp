import objax
import jax.numpy as np

class batch():
    def __init__(self, func):
        self.func = func
        self.variable = self.func.__name__
        self.original_getter = None


    def getter(self, obj):
        return getattr(obj, f'raw_{self.variable}').value

    def __get__(self, obj, objtype=None):
        return self.func(self, lambda: self.getter(obj))

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

