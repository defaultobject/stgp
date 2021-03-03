import gpjax
import numpy as np
import jax.numpy as jnp


X = np.linspace(0, 1, 100)
Y = np.sin(X)

m = gpjax.model.GP()
print(m)
m = gpjax.model.GP(gpjax.model.GP())
print(m)
