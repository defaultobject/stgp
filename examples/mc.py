import pods
import numpy as np
import GPy
import matplotlib.pyplot as plt

import urllib

from deepgp_tutorial import initialize
from deepgp_tutorial import staged_optimize
from deepgp_tutorial import posterior_sample
from deepgp_tutorial import visualize
from deepgp_tutorial import visualize_pinball

import sys
sys.path.append('/Users/ohamelijnck/Documents/code/PyDeepGP')
import deepgp
deepgp.DeepGP.initialize=initialize
deepgp.DeepGP.staged_optimize=staged_optimize
deepgp.DeepGP.posterior_sample=posterior_sample
deepgp.DeepGP.visualize=visualize
deepgp.DeepGP.visualize_pinball=visualize_pinball

data = pods.datasets.mcycle()
x = data['X']
y = data['Y']

scale=np.sqrt(y.var())
offset=y.mean()
yhat = (y - offset)/scale


layers = [y.shape[1], 1, x.shape[1]]
inits = ['PCA']*(len(layers)-1)
kernels = []
for i in layers[1:]:
    kernels += [GPy.kern.RBF(i)]

m = deepgp.DeepGP(
    layers,Y=yhat, X=x, 
    inits=inits, 
    kernels=kernels, # the kernels for each layer
    num_inducing=20, 
    back_constraint=False
)

m.initialize()

m.predict(x)

breakpoint()

m.staged_optimize(iters=(1000,1000,10000), messages=(True, True, True))

xs = np.linspace(-20, 80, 500)[:, None]

mu, var = m.predict(xs)

plt.fill_between(np.squeeze(xs), np.squeeze(mu-2*np.sqrt(var)), np.squeeze(mu+2*np.sqrt(var)), alpha=0.4)
plt.plot(xs, mu)
plt.show()


breakpoint()
