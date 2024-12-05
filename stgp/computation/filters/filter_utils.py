from ...dispatch import _ensure_str

def _get_data_or_sparsity_spatial_points(data, sparsity):
    if _ensure_str( sparsity) == 'FITCSpatialSparsity':
        return sparsity.raw_Z.X_space
    else:
        return data.X_space

def _get_prior_spatial_points(data, prior):
    sparsity = prior.base_prior.get_sparsity()

    # TODO: this is a hack so we don't break all existing code to handle uncertain inputs
    # at some point refactor as it would be nice if we supported multiple latent functiosn 
    # with different spatial inducing point locations
    if _ensure_str(prior) == 'UncertainPredictionInput':
        return [
            _get_data_or_sparsity_spatial_points(data, sparsity[i])
            for i in range(len(sparsity))
        ]
    else:
        # assuming that all latents have the same spatial points
        sparsity = sparsity[0]
        return _get_data_or_sparsity_spatial_points(data, sparsity)

