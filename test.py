from gpjax.kernels import Matern32, RBF, ApproximateMarkovKernel
import numpy as np

X = np.random.rand(20, 3)

k1 = Matern32(input_dim=1, active_dims=[0], variance=2.0)
k3 = Matern32(input_dim=1, active_dims=[1])

k = k1*k1*k3

print(k[:])
print(k[0])
print(k[1:])
#k.K(X, X)
#print(k.lengthscales)
#print(k.variances)
