import legogp as lego
import numpy as np
import pandas as pd

# generate data
P = 10

x = np.linspace(0, 1, 100)
y = np.sin(x*10)

X = x[:, None]
Y = y[:, None]

m = lego.models.GP(X, Y)

print(m.get_objective())
