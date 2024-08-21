from . import LinearTransform

class UncertainPredictionInput(LinearTransform):
    """ Using an UncertainPredictionInput makes no difference in training, only in prediction """
    
    def __init__(self, base_gp, prediction_gp, active_dim):
        self.base_gp = base_gp
        self.prediction_gp = prediction_gp
        self.active_dim = active_dim
        self._input_dim = 1
        self._output_dim = 1
        self._parent = self.base_gp

    def covar(self, X1, X2):
        return self.parent.covar(X1, X2)
    def mean(self, X1):
        return self.parent.mean(X1)

    @property
    def approximately_linear(self) -> bool:
        return True

