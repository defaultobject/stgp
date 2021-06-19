import jax.numpy as np
import objax
from ..dispatch import evoke
from ..integral_approximators.factory import get_approximator
from .. import settings
import chex

class ApproximatePosterior(objax.Module):
    def __init__(self, whiten=False):
        self.whiten = whiten

        self.approximator = get_approximator()

        self.generator = objax.random.Generator(seed=0)

    def ELL(self, X: np.ndarray, Y: np.ndarray, likelihood: 'Likelihood', kernel: 'Kernel', sparsity: 'Sparsity', minibatch):

        ell_fn = evoke('expected_log_likelihood')

        N = X.shape[0]

        if minibatch is not None:
            #minibatch
            idx = objax.random.randint((minibatch,), low=0, high=N-1, generator=self.generator)

            X = X[idx,:]
            Y = Y[idx,:]

            print(idx, Y)

        else:
            minibatch = N

        try:
            if settings.force_black_box:
                raise NotImplementedError()

            ell = ell_fn(
                X,
                Y,
                self, 
                likelihood,
                kernel,
                sparsity
            )
        except NotImplementedError as e:
            #use black box inference
            ell = self.approximator.run(
                X,
                Y,
                self, 
                likelihood,
                kernel,
                sparsity
            )

            ell = np.sum(ell)

        print('ell: ', ell)
        ell = (N/minibatch)*ell
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

    def predictive_marginal(self, XS: np.ndarray, X: np.ndarray, kernel: 'Kernel', sparsity: 'Sparsity', diagonal=True):
        if diagonal:
            if self.whiten:
                marginal_fn = evoke('whitened_diagonal_marginal')
            else:
                marginal_fn = evoke('diagonal_marginal')
        else:
            if self.whiten:
                marginal_fn = evoke('whitened_full_marginal')
            else:
                marginal_fn = evoke('full_marginal')

        m, S = marginal_fn(
            XS, 
            X,
            self, 
            kernel,
            sparsity
        )

        return m, S

    def predict(self, XS: np.ndarray, X: np.ndarray, likelihood: 'Likelihood', kernel: 'Kernel', sparsity, diagonal):
        if diagonal:
            predict_fn = evoke('predict_diagonal')
        else:
            predict_fn = evoke('predict_full')

        try:
            if settings.force_black_box:
                raise NotImplementedError()

            mean, var = predict_fn(
                XS,
                X,
                self, 
                likelihood,
                kernel,
                sparsity
            )

        except NotImplementedError as e:
            #use black box inference
            mean, var = self.approximator.predict_run(
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
    
