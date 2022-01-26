import objax
class MultiObjectiveModel(objax.Module):
    def __init__(self, model_list):
        self.model_list = objax.ModuleList(model_list)

    def get_objective(self):
        obj = 0.0
        for m in self.model_list:
            obj += m.get_objective()

        return obj


