import jax

from ..sparsity import NoSparsity
from ..models import GP
from ..transforms.multi_output import LMC, GPRN, GPRN_Exp
from ..data import Data
from ..likelihood import Gaussian

def lmc_regression(X, Y, P, Q, kernels, inference):

    Z = [NoSparsity(X) for q in range(Q)]

    latent_gps = [
        GP(sparsity=Z[q], kernel=kernels[q]) for q in range(Q)
    ] 

    if inference == 'vi':
        inference='Variational'

    # Construct LMC Prior
    prior = LMC(latent_gps, output_dim = P)

    data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=0.1) for p in range(P)],
        prior=prior,
        inference=inference,
    )

    return m

def gprn_regression(X, Y, P, Q, W_kernels, f_kernels, inference, constraint=None):
    Z_f = [NoSparsity(X) for q in range(Q)]
    Z_W = [[NoSparsity(X) for q in range(Q)] for p in range(P)]

    latent_f_gps = [
        GP(sparsity=Z_f[q], kernel=f_kernels[q]) for q in range(Q)
    ] 

    latent_W_gps = [
        [GP(sparsity=Z_W[p][q], kernel=W_kernels[p][q]) for q in range(Q)]
        for p in range(P)
    ] 

    if inference == 'vi':
        inference='Variational'

    # Construct LMC Prior
    if constraint is None:
        prior = GPRN(latent_W_gps, latent_f_gps, output_dim = P)
    elif constraint is 'exp':
        prior = GPRN_Exp(latent_W_gps, latent_f_gps, output_dim = P)
    else:
        raise NotImplementedError()

    data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=0.1) for p in range(P)],
        prior=prior,
        inference=inference,
        ell_samples=100,
        prediction_samples=100
    )

    return m



