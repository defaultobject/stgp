from gpjax.kernels import Matern32, RBF, ApproximateMarkovKernel
import numpy as np

X = np.random.rand(20, 3)

k1 = Matern32(input_dim=1, active_dim=0, variance=2.0)
k2 = ApproximateMarkovKernel(RBF(input_dim=2, active_dim=1))
k3 = Matern32(input_dim=1, active_dim=1)

k = k1*k2*k3
print(k.vars())
#print(k.lengthscales)
#print(k.variances)
