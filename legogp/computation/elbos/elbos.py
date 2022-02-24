import jax
import jax.numpy as np
from jax import jit
import objax
import chex

from ...utils.utils import can_batch
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior
from ...transforms import Independent, Transform
from ...likelihood import ProductLikelihood
from ...dispatch import dispatch, evoke
from .expected_log_likelihoods import precomputed_expected_log_likelihood 
from ..marginals import diagonal_marginal, whitened_diagonal_marginal

from .prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ

@dispatch(ProductLikelihood, Transform, MeanFieldApproximatePosterior)
def elbo(
    X: np.ndarray, Y: np.ndarray, likelihood: ProductLikelihood, prior: Independent, approximate_posterior: MeanFieldApproximatePosterior, inference: 'Variational'
):
    N = Y.shape[0]

    # Compute KL term
    # TODO: this should not have X
    KL = evoke('kullback_leibler', approximate_posterior, prior)(
        X, approximate_posterior, prior
    )

    # Minibatching across all outputs
    minibatch = False
    if inference.minibatch_size is not None:
        minibatch = True

    if minibatch:
        #minibatch
        minibatch_size = inference.minibatch_size

        idx = objax.random.randint((minibatch_size,), low=0, high=N-1, generator=inference.generator)

        X = X[idx,:]
        Y = Y[idx,:]

        # Compute approximate posterior
        # TODO: this need to depend on sparisty and if whitened or not
        q_f_mu, q_f_var = evoke('marginal', 'prediction', approximate_posterior, prior)(
            X, X, approximate_posterior, prior
        )

    else:
        minibatch_size = N

        q_f_mu, q_f_var = evoke('marginal', approximate_posterior, prior)(
            X, approximate_posterior, prior
        )


    # Compute Expected Log Likelihood   
    ELL = evoke('expected_log_likelihood', likelihood, prior, approximate_posterior)(
        X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference
    )

    breakpoint()
    # TODO: the minibatch scaling is biased when there is missing data
    return (N/minibatch_size) * ELL - KL
    #return  ELL 
