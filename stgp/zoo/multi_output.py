import jax
import jax.numpy as np

from ..sparsity import NoSparsity
from ..models import GP
from ..transforms.multi_output import LMC, LMC_DRD, GPRN, GPRN_Exp, GPRN_DRD, GPRN_DRD_EXP
from ..data import Data, TransformedData
from ..likelihood import Gaussian
from ..kernels import RBF, ScaleKernel
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
from stgp.transforms.basic import Log, Softminus, Affine, ReverseFlow

def lmc_regression(X, Y, P=None, Q=None, kernels = None, inference = 'Batch', lengthscale=1.0, variance=1.0, lik_noise = 0.1, normalise_data: bool = False):
    """
    Helper function for returning an LMC model with Gaussian likelihood across all outputs.

    Args:
        P, Q: When not passed they are set to Y.shape[1]
        kernels: When not passed they are set to RBF kernels
    """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    if Q is None:
        Q = P

    if kernels is None:
        kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                variance = variance
            )
            for q in range(Q)
        ]


    # construct model
    Z = [NoSparsity(X) for q in range(Q)]

    latent_gps = [
        GP(sparsity=Z[q], kernel=kernels[q]) for q in range(Q)
    ] 

    # Construct LMC Prior
    prior = LMC(latent_gps, output_dim = P)

    if inference == 'vi':
        inference='Variational'
        approximate_posterior = FullGaussianApproximatePosterior(
            dim = X.shape[0]*prior.base_prior.output_dim
        ) 
    else:
        approximate_posterior = None



    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference=inference,
        approximate_posterior = approximate_posterior
    )

    return m

def lmc_drd_regression(X, Y, P=None, Q=None, kernels = None, inference = 'Batch', lengthscale=1.0, variance=1.0, lik_noise = 0.1, normalise_data: bool = False):
    """
    Helper function for returning an LMC-DRD model with Gaussian likelihood across all outputs.

    Args:
        P, Q: When not passed they are set to Y.shape[1]
        kernels: When not passed they are set to RBF kernels
    """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    if Q is None:
        Q = P

    if kernels is None:
        # DRD should not have a scale kernel 
        kernels = [
            RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale)
            for q in range(Q)
        ]

    # construct model
    Z = [NoSparsity(X) for q in range(Q)]

    latent_gps = [
        GP(sparsity=Z[q], kernel=kernels[q]) for q in range(Q)
    ] 

    prior = LMC_DRD(latent_gps, output_dim = P)

    if inference == 'vi':
        inference='Variational'
        approximate_posterior = FullGaussianApproximatePosterior(
            dim = X.shape[0]*prior.base_prior.output_dim
        ) 
    else:
        approximate_posterior = None


    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference=inference,
        approximate_posterior = approximate_posterior
    )

    return m


def gprn_regression(X, Y, P=None, Q=None, W_kernels=None, f_kernels=None, inference='Variational', constraint=None, ell_samples=100, lengthscale=1.0, variance=1.0, lik_noise=0.1, normalise_data=True):
    """ Helper function for returning a variational mean-field GPRN model with Gaussian likelihood across all outputs.  """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    if Q is None:
        Q = P

    if W_kernels is None:
        W_kernels = [
            [
                ScaleKernel(
                    RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                    variance = variance
                )
                for q in range(Q)
            ]
            for p in range(P)
        ]

    if f_kernels is None:
        f_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                variance = variance
            )
            for q in range(Q)
        ]

    if inference == 'vi':
        inference='Variational'

    # setup model

    Z_f = [NoSparsity(X) for q in range(Q)]
    Z_W = [[NoSparsity(X) for q in range(Q)] for p in range(P)]

    latent_f_gps = [
        GP(sparsity=Z_f[q], kernel=f_kernels[q]) for q in range(Q)
    ] 

    latent_W_gps = [
        [GP(sparsity=Z_W[p][q], kernel=W_kernels[p][q]) for q in range(Q)]
        for p in range(P)
    ] 

    # Construct LMC Prior
    if constraint is None:
        prior = GPRN(latent_W_gps, latent_f_gps, output_dim = P)
    elif constraint == 'exp':
        prior = GPRN_Exp(latent_W_gps, latent_f_gps, output_dim = P)
    else:
        raise NotImplementedError()

    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference=inference,
        ell_samples=ell_samples,
        prediction_samples=None
    )

    return m


def gprn_drd_regression(X, Y, P=None, W_kernels=None, f_kernels=None, latent_variance = 1.0, variance = 1.0, ell_samples=100, lengthscale=1.0,  lik_noise=0.1, meanfield=True, normalise_data=True):
    """ Helper function for returning a variational full-Gaussian GPRN_DRD model with Gaussian likelihood across all outputs.  """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    Q = P

    num_W = int(Q * (Q-1)/2)

    if W_kernels is None:
        W_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                variance = variance
            )
            for q in range(num_W)
        ]

    if f_kernels is None:
        # kernel variances must be 1, so that K is a correlation matrix
        f_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                variance = variance
            )
            for q in range(Q)
        ]

        # fix f kernels variance
        [kern.variance_param.fix() for kern in f_kernels]

    # setup model

    Z_f = [NoSparsity(X) for q in range(Q)]
    Z_W = [NoSparsity(X) for q in range(num_W)]

    latent_f_gps = [
        GP(sparsity=Z_f[q], kernel=f_kernels[q]) for q in range(Q)
    ] 

    latent_W_gps = [
        GP(sparsity=Z_W[q], kernel=W_kernels[q]) for q in range(num_W)
    ] 


    prior = GPRN_DRD(
        latent_W_gps, 
        latent_f_gps,
        input_dim = Q,
        output_dim = P,
        variances = np.ones(P)*latent_variance
    )

    if meanfield:
        q = None
    else:
        q = FullGaussianApproximatePosterior(
            dim = X.shape[0]*prior.base_prior.output_dim
        )

    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference='Variational',
        approximate_posterior = q,
        ell_samples=ell_samples,
        prediction_samples=None
    )

    return m


def gprn_drd_nv_regression(X, Y, P=None, W_kernels=None, f_kernels=None, v_kernels=None, latent_variance = 1.0, variance = 1.0, ell_samples=100, lengthscale=1.0,  lik_noise=0.1, meanfield=True, normalise_data=True):
    """ Helper function for returning a variational full-Gaussian Noise Varying GPRN_DRD model with Gaussian likelihood across all outputs.  """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    Q = P

    num_W = int(Q * (Q-1)/2)

    if v_kernels is None:
        v_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                variance = variance
            )
            for q in range(P)
        ]

    if W_kernels is None:
        W_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                variance = variance
            )
            for q in range(num_W)
        ]

    if f_kernels is None:
        # kernel variances must be 1, so that K is a correlation matrix
        f_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale),
                variance = variance
            )
            for q in range(Q)
        ]

        # fix f kernels variance
        [kern.variance_param.fix() for kern in f_kernels]

    # setup model

    Z_v = [NoSparsity(X) for p in range(P)]
    Z_f = [NoSparsity(X) for q in range(Q)]
    Z_W = [NoSparsity(X) for q in range(num_W)]

    latent_v_gps = [
        GP(sparsity=Z_v[q], kernel=v_kernels[q]) for q in range(P)
    ] 

    latent_f_gps = [
        GP(sparsity=Z_f[q], kernel=f_kernels[q]) for q in range(Q)
    ] 

    latent_W_gps = [
        GP(sparsity=Z_W[q], kernel=W_kernels[q]) for q in range(num_W)
    ] 


    prior = GPRN_DRD_EXP(
        latent_v_gps, 
        latent_W_gps, 
        latent_f_gps,
        input_dim = Q,
        output_dim = P,
        variances = np.ones(P)*latent_variance
    )

    if meanfield:
        q = None
    else:
        q = FullGaussianApproximatePosterior(
            dim = X.shape[0]*prior.base_prior.output_dim
        )

    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference='Variational',
        approximate_posterior = q,
        ell_samples=ell_samples,
        prediction_samples=None
    )

    return m
