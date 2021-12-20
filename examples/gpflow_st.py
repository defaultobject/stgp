import gpflow
from gpflow import set_trainable
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_grid(x1, x2, y1, y2, n1=10, n2=10):
    y = np.linspace(y1, y2, n2)
    x = np.linspace(x1, x2, n1)

    grid = []
    for i in x:
        for j in y:
            grid.append([i, j])

    return np.array(grid)


Nt_train = 10
Ns = 10

X = create_grid(-1, 1, -1, 1, Nt_train, Ns)
N = X.shape[0]

y = np.sin(10*X[:, 0]) + np.sin(10*X[:, 1]) + 0.01*np.random.randn(N)
Y = y[:, None]

k = gpflow.kernels.RBF(
    lengthscales = [0.1, 0.1]
)
m = gpflow.models.GPR(data=(X, Y), kernel=k)

opt = tf.optimizers.Adam(0.01)
learning_curve = []
epochs = 100

for i in tqdm(range(epochs)):
    opt.minimize(
        m.training_loss, 
        var_list=m.trainable_variables
    )
    likelihood = -m.log_marginal_likelihood()
    learning_curve.append(likelihood)

plt.plot(learning_curve)
plt.show()
