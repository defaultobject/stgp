import jax
import jax.numpy as np
import chex

from ...dispatch import evoke, dispatch

# Import types
from ...transforms import Transform, LinearTransform, NonLinearTransform
from ...likelihood import ProductLikelihood
from ...data import Data, TransformedData
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo

@dispatch(Data, 'BatchGP', ProductLikelihood, NonLinearTransform)
@dispatch(Data, 'VGP', ProductLikelihood, NonLinearTransform)
def confidence_intervals(XS, m):
    latents = m.prior.latent_obj

    latent_mu, latent_var =  evoke('marginal', 'prediction', m.approximate_posterior, m.likelihood, latents)(
        XS, m.data, m.approximate_posterior, m.likelihood, latents, m.inference, True
    ) 

    vmaped_prior_forard =  jax.vmap(m.prior.forward, [1], 0)

    mu = mv_indepentdent_monte_carlo(
        lambda f, fn: fn(f),
        latent_mu,
        latent_var,
        fn_args=[vmaped_prior_forard],
        generator = m.inference.generator, 
        num_samples = m.inference.prediction_samples,
        average=False
    )

    # Ensure correct shape
    mu = np.reshape(mu, [m.inference.prediction_samples, XS.shape[0], m.prior.output_dim])

    ci_lower = jax.numpy.percentile(mu, 2.5, axis=0)
    median = jax.numpy.percentile(mu, 50, axis=0)
    ci_upper = jax.numpy.percentile(mu, 97.5, axis=0)

    # ensure shape is [P, N]
    return median.T, ci_lower.T, ci_upper.T

@dispatch(Data, 'BatchGP', ProductLikelihood, LinearTransform)
@dispatch(Data, 'VGP', ProductLikelihood, LinearTransform)
def confidence_intervals(XS, m):
    mu, var = m.predict_y(XS, squeeze=False, diagonal=True)

    P = mu.shape[0]

    # Ensure rank 2
    mu = np.reshape(mu, [P, -1])
    var = np.reshape(var, [P, -1])

    return mu, mu-1.96*np.sqrt(var), mu+1.96*np.sqrt(var)

@dispatch(TransformedData, 'BatchGP', ProductLikelihood, Transform)
@dispatch(TransformedData, 'VGP', ProductLikelihood, Transform)
def confidence_intervals(XS, m):
    base_data = m.data.base_data

    median, lower_ci, upper_ci = evoke(
        'confidence_intervals', base_data, m, m.likelihood, m.prior         
    )(XS, m)

    median = m.data.inverse_transform(median.T).T
    lower_ci = m.data.inverse_transform(lower_ci.T).T
    upper_ci = m.data.inverse_transform(upper_ci.T).T

    return median, lower_ci, upper_ci

@dispatch('BatchGP')
@dispatch('VGP')
def confidence_intervals(XS, m):
    return evoke(
        'confidence_intervals', m.data, m, m.likelihood, m.prior         
    )(XS, m)
