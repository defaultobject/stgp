import jax.numpy as np
import objax
from ..dispatch import evoke
import chex

class ApproximatePosterior(objax.Module):
    def ELL(self, X: np.ndarray, Y: np.ndarray, likelihood: 'Likelihood', kernel: 'Kernel', sparsity: 'Sparsity'):

        ell_fn = evoke('expected_log_likelihood')

        if ell_fn is None:
            #The ELL term must be black boxed
            raise NotImplementedError()
        else:
            ell = ell_fn(
                X,
                Y,
                self, 
                likelihood,
                kernel,
                sparsity
            )

        chex.assert_rank(ell, 0)

        return ell

    def marginal(self, X: np.ndarray,  kernel: 'Kernel', sparsity: 'Sparsity'):
        marginal_fn = evoke('diagonal_marginal')

        if marginal_fn is None:
            raise NotImplementedError()
        else:
            m, S_diag = marginal_fn(
                X,
                self, 
                kernel,
                sparsity
            )

        return m, S_diag

    def predictive_marginal(self, XS: np.ndarray, X: np.ndarray, kernel: 'Kernel', sparsity: 'Sparsity'):
        marginal_fn = evoke('diagonal_marginal')

        if marginal_fn is None:
            raise NotImplementedError()
        else:
            m, S_diag = marginal_fn(
                XS, 
                X,
                self, 
                kernel,
                sparsity
            )

        return m, S_diag

    def predict(self, XS: np.ndarray, X: np.ndarray, likelihood: 'Likelihood', kernel: 'Kernel', sparsity):
        predict_fn = evoke('predict_diagonal')

        if predict_fn is None:
            raise NotImplementedError()
        else:
            mean, var = predict_fn(
                XS,
                X,
                self, 
                likelihood,
                kernel,
                sparsity
            )

        return mean, var

    def KL(self, X: np.ndarray, kernel: 'Kernel', sparsity: 'Sparsity'):
        kl_fn = evoke('KL')

        if kl_fn is None:
            raise NotImplementedError()
        else:
            kl = kl_fn(
                X,
                self, 
                kernel,
                sparsity
            )

        chex.assert_rank(kl, 0)

        return kl
    
