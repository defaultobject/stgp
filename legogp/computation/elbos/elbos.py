import jax
import jax.numpy as np
from jax import jit
import objax
import chex

from batchjax import batch_or_loop
from ...utils.utils import get_batch_type
from ...approximate_posteriors import ApproximatePosterior, ConjugateApproximatePosterior, FullConjugateGaussian
from ...transforms import Independent, Transform
from ...likelihood import Likelihood
from ...dispatch import dispatch, evoke

from .prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ

# TODO: this needs to be written as a function of q_z_mu, q_z_var
# Just to make it easier to compute gradients wrt to them :) 
def compute_expected_log_liklihood(X, Y, likelihood, prior, approximate_posterior, inference):
    N = Y.shape[0]

    # Minibatching across all outputs
    minibatch = False
    if inference.minibatch_size is not None:
        minibatch = True

    if minibatch:
        #minibatch
        minibatch_size = inference.minibatch_size

        # TODO: minibatching should only happen at locations WITHOUT missing data
        # TODO: OR scaling should take into account the missing data
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

        q_f_mu, q_f_var = evoke('marginal', approximate_posterior, likelihood, prior)(
            X, approximate_posterior, likelihood, prior
        )


    # Compute Expected Log Likelihood   
    ELL = evoke('expected_log_likelihood', likelihood, prior, approximate_posterior)(
        X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference
    )

    return (N/minibatch_size) * ELL

@dispatch(Likelihood, Transform, ApproximatePosterior)
def elbo(
    X: np.ndarray, Y: np.ndarray, likelihood: Likelihood, prior: Independent, approximate_posterior: ApproximatePosterior, inference: 'Variational'
):
    N = Y.shape[0]

    # Compute KL term
    KL = evoke('kullback_leibler', approximate_posterior, prior)(
        approximate_posterior, prior
    )

    # Compute expected log likelihood term
    ELL = compute_expected_log_liklihood(X, Y, likelihood, prior, approximate_posterior, inference)

    return  ELL - KL
    #return  ELL


@dispatch(Likelihood, Transform, ConjugateApproximatePosterior)
def elbo(
    X: np.ndarray, Y: np.ndarray, likelihood: Likelihood, prior: Transform, q: ConjugateApproximatePosterior, inference: 'Variational'
):
    # Compute ELL
    ELL = compute_expected_log_liklihood(X, Y, likelihood, prior, q, inference)

    # Compute surrogate ELL
    # TODO: this needs to be generalised to map across all X
    ELL_surrogate = compute_expected_log_liklihood(
        q.X[0], # ALL X has to be the same so this makes no difference
        q.Y, 
        q.likelihood, 
        prior, 
        q, 
        inference
    )

    # Compute surrogate marginal likelihood
    # TODO: assuming a mean-field approx posterior
    q_list = q.approx_posteriors

    # get_ojective returns the negative log liklihood
    # We require the (postive) log liklihood
    ML_arr =  batch_or_loop(
        lambda qq: -qq.surrogate.get_objective(),
        [q_list],
        [0],
        dim=len(q_list),
        out_dim = 1,
        batch_type = get_batch_type(q_list)
    )

    ML_surrogate = np.sum(ML_arr)

    return ELL - ELL_surrogate + ML_surrogate


@dispatch(Likelihood, Transform, FullConjugateGaussian)
def elbo(
    X: np.ndarray, Y: np.ndarray, likelihood: Likelihood, prior: Transform, q: ConjugateApproximatePosterior, inference: 'Variational'
):
    Q = prior.num_latents

    X = np.tile(X, [Q, 1, 1])

    # Compute ELL
    ELL = compute_expected_log_liklihood(X, Y, likelihood, prior, q, inference)

    # Compute surrogate ELL
    ELL_surrogate = compute_expected_log_liklihood(
        q.X, 
        q.Y, 
        q.likelihood, 
        prior, 
        q, 
        inference
    )
    ML_surrogate = - q.surrogate.get_objective()

    elbo =  ELL - ELL_surrogate + ML_surrogate
    #elbo =  ELL 

    return elbo

