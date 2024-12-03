from ...dispatch import _ensure_str

def _get_prior_spatial_points(data, prior):
    sparsity = prior.base_prior.get_sparsity()[0]
    # TODO: dispatch over everything
    if _ensure_str( sparsity) == 'FITCSpatialSparsity':
        return sparsity.raw_Z.X_space
    else:
        return data.X_space
