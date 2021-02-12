from ..parameter import Parameter

from . import SpatialSparsity

import jax
import jax.numpy as np

from typing import Optional

class SpatialStructuredSparsity(SpatialSparsity):
    def __init__(self, inducing_locations: Optional[np.ndarray]=None, temporal_locations:Optional[np.ndarray]=None, trainable: Optional[bool]=True, name: Optional[str]='spatial_sparsity'):
        self.name = name

        self.temporal_locations = temporal_locations

        super(SpatialStructuredSparsity, self).__init__(inducing_locations, trainable, name)

    def get_Z_across_time(self, temporal_locations):
        Z  = self.inducing_locations.val
        temporal_locations= np.array(temporal_locations)

        temporal_locations = np.expand_dims(temporal_locations, -1)
        temporal_locations = np.expand_dims(temporal_locations, -1)

        temporal_locations = np.tile(temporal_locations, (1, Z.shape[0], 1))

        def concat(time, Z):
            return np.hstack([time, Z[:, 1:]])

        tiled_Z_across_time = jax.vmap(concat, (0, None), 0)(temporal_locations, Z)
        return tiled_Z_across_time.reshape([-1, Z.shape[1]])

    @property
    def Z(self):
        return self.get_Z_across_time(self.temporal_locations)




