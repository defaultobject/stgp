from . import LinearTransform

class UncertainPredictionInput(LinearTransform):
    """ Using an UncertainPredictionInput makes no difference in training, only in prediction """
    
    def __init__(self, prior, prediction_gp, active_dim=None):
        self.prior = prior
        self.prediction_gp = prediction_gp
        self.active_dim = active_dim
        self._input_dim = 1
        self._output_dim = 1
        self._parent = self.prior

    def covar(self, X1, X2):
        return self.parent.covar(X1, X2)
    def mean(self, X1):
        return self.parent.mean(X1)

    @property
    def approximately_linear(self) -> bool:
        return True

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def m_inf(self, x, X_s, t):
        return self.parent.m_inf(x, X_s, t)

    def H(self, x, X_s, t):
        return self.parent.H(x, X_s, t)


    

