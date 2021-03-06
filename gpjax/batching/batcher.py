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

