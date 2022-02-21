from ...dispatch import dispatch, evoke
from .kullback_leiblers import gaussian_kl
from ...transforms import LinearTransform, Independent

@dispatch('MeanFieldApproximatePosterior', Independent)
def kullback_leibler(X, approximate_posterior, prior):
    breakpoint()
    pass


@dispatch('MeanFieldApproximatePosterior', LinearTransform)
def kullback_leibler(X, approximate_posterior, prior):
    latents = prior.latents

    return evoke('kullback_leibler', approximate_posterior, latents)(
        X, approximate_posterior, latents
    )


