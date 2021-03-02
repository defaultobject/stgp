import gpjax
import numpy as np


X = np.linspace(0, 1, 100)
Y = np.sin(X)

gpjax.model.GP(X, Y)
