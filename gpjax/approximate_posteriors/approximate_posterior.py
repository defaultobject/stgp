import jax.numpy as np
import objax
from ..dispatch import evoke
import chex

class ApproximatePosterior(objax.Module):
    def __init__(self, whiten=False):
        self.whiten = whiten

    def ELL(self, X: np.ndarray, Y: np.ndarray, likelihood: 'Likelihood', kernel: 'Kernel', sparsity: 'Sparsity'):

        ell_fn = evoke('expected_log_likelihood')

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
        if self.whiten:
            marginal_fn = evoke('whitened_diagonal_marginal')
        else:
            marginal_fn = evoke('diagonal_marginal')

        m, S_diag = marginal_fn(
            X,
            self, 
            kernel,
            sparsity
        )

        return m, S_diag

    def predictive_marginal(self, XS: np.ndarray, X: np.ndarray, kernel: 'Kernel', sparsity: 'Sparsity'):
        if self.whiten:
            marginal_fn = evoke('whitened_diagonal_marginal')
        else:
            marginal_fn = evoke('diagonal_marginal')

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
        if self.whiten:
            kl_fn = evoke('whitened_KL')
        else:
            kl_fn = evoke('KL')


        kl = kl_fn(
            X,
            self, 
            kernel,
            sparsity
        )

        chex.assert_rank(kl, 0)

        return kl
    
