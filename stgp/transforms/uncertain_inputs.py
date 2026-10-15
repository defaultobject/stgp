import jax
import objax

from . import LinearTransform

class UncertainPredictionInput(LinearTransform):
    """Use lower-fidelity posterior predictions as uncertain inputs."""
    
    def __init__(
        self,
        prior,
        prediction_gp=None,
        approximation="layerwise",
        state_input_indices=None,
        active_dim=None,
        cache_training_predictions=False,
    ):
        self.prior = prior
        self.prediction_gp = None if prediction_gp is None else objax.ModuleList(prediction_gp)
        self.approximation = approximation
        self.state_input_indices = state_input_indices

        if approximation not in {"layerwise", "frozen_weights"}:
            raise ValueError(f"Unknown uncertain-input approximation: {approximation}")
        if approximation == "frozen_weights":
            if prediction_gp is not None:
                raise ValueError("frozen_weights obtains its inputs from the joint state")
            if state_input_indices is None:
                raise ValueError("frozen_weights requires state_input_indices")
        elif prediction_gp is None:
            raise ValueError("layerwise uncertain inputs require prediction_gp")
        self.active_dim = active_dim
        self._input_dim = 1
        self._output_dim = 1
        self._parent = self.prior
        self.training_prediction_cache = None

        if cache_training_predictions:
            self.refresh_training_prediction_cache()

    def refresh_training_prediction_cache(self):
        """Cache detached LF posteriors for high-fidelity training."""
        if self.approximation == "frozen_weights":
            return
        self.training_prediction_cache = []

        for gp in self.prediction_gp:
            if gp is None:
                self.training_prediction_cache.append(None)
                continue

            pred_mu, pred_var = gp.posterior_in_latent_data()
            if pred_mu.shape[0] > 1:
                pred_mu = pred_mu[0]
                pred_var = pred_var[0]

            self.training_prediction_cache.append((
                jax.lax.stop_gradient(pred_mu),
                jax.lax.stop_gradient(pred_var),
            ))

    def clear_training_prediction_cache(self):
        self.training_prediction_cache = None

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



    
