import objax

from .computation.parameter_transforms import inv_positive_transform, positive_transform


class Parameter(objax.Module):
    """
    A wrapper around objax Trainvar to support naming a Parameter (for pretty printing of models)
        and to make constraints easier to use.
    """
    _NAME_DICT = {}

    def __init__(self, val, constraint:str=None, name=None):
        self.constraint = constraint
        self.raw_var = objax.TrainVar(self.inv_transform(val))

        self.set_name(name)

    def set_name(self, name):
        if name is not None:
            if name in Parameter._NAME_DICT:
                Parameter._NAME_DICT[name] += 1
                self.name = f'{name} - {Parameter._NAME_DICT[name]}'
            else:
                Parameter._NAME_DICT[name] = 1
                self.name = name
        else:
            self.name = None
    
    @property
    def value(self):
        return self.transform(self.raw_var.value)
    
    def transform(self, var):
        if self.constraint == None:
            return var
        elif self.constraint == 'positive':
            return positive_transform(var)

        raise RuntimeError(f'Constraint {self.constraint} is not supported!')

    def inv_transform(self, val):

        if self.constraint == None:
            return val
        elif self.constraint == 'positive':
            return inv_positive_transform(val)

        raise RuntimeError(f'Constraint {self.constraint} is not supported!')

