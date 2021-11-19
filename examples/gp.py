import legogp as lego
import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import stdata as st
from stdata.plots import grid_to_matrix
import matplotlib.pyplot as plt

# generate data
P = 10

x = np.linspace(0, 1, 100)
y = np.sin(x*10)

X = x[:, None]
Y = y[:, None]

print(X.shape, Y.shape)

layer1 = [lego.models.GP(X), lego.models.GP(X)]

m2 = lego.models.GP(
    X, Y,
    kernel=lego.kernels.deep_kernels.DeepRBF(layer1[0])
)

plt.imshow(np.array(m2.covar(X, X)[0]))
plt.show()
