from . import Distribution
from ..parameter import Parameter
from ..likelihoods import *
from ..computation import *
from ..computation.general import log_chol_matrix_det, cholesky_solve
from ..computation.exponential_family import *
from ..settings import Settings

from .. import Kernel

import jax
import jax.numpy as np
from jax import jit, partial

from ..decorators import return_gradients

import typing
from typing import Optional, Union, List


class GaussianDistribution(Distribution):
    """
    A Gaussian distribution parameterised by (m , S)
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        covariance_chol: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        trainable: Optional[bool] = True,
    ) -> None:
        self.meta = meta

        if self.meta is None:
            self.meta = {}

        self.name = name
        if self.name is None:
            self.name = "GaussianDistribution"

        super(GaussianDistribution, self).__init__(self.name, self.meta, trainable)

        scope = "variational"

        self.mu = mu
        if self.mu is not None:
            self.mu = self.parameter(
                val=self.mu,
                scope=scope,
                train=trainable,
                module_name=self.name,
                param_name="mu",
            )

        self.covariance_chol = covariance_chol
        if self.covariance_chol is not None:
            self.covariance_chol = self.parameter(
                val=self.covariance_chol,
                scope=scope,
                meta={"N": self.meta["N"]},
                constraint="lower triangular",
                train=trainable,
                module_name=self.name,
                param_name="covariance_chol",
            )

        self.key = jax.random.PRNGKey(0)

    def get_default_params(self):
        self.meta["N"] = 1

        return {"mean": np.array(1.0), "covariance_chol": np.array(1.0)}

    def mean(self, X=None):
        return self.mu.value

    # @jit
    def covar(self, X1, X2):
        covar_sqrt = self.covar_chol(X1, X2)
        return np.matmul(covar_sqrt, covar_sqrt.T)

    def covar_diag(self, X1):
        return np.diag(self.covar(X1, X1))

    def covar_chol(self, X1, X2):
        return self.covariance_chol.value

        #    v = self.covar(X1, X2)
        #    v = v + Settings.jitter*np.eye(v.shape[0])
        #    return jax.scipy.linalg.cholesky(v, lower=True)

    def predict(self, X, full_covar=False):
        if full_covar:
            return self.mean(X), self.covar(X, X)

        return self.mean(X), np.diag(self.covar(X, X))

    @return_gradients
    def KL(self, distribution=None, X=None):
        if distribution is None:
            raise RuntimeError("KL: distribution must be specified")

        for key, func in self.kl_dict.items():
            if isinstance(distribution, key):
                mu_1 = self.mean(X)
                sigma_chol_1 = self.covar_chol(X, X)

                mu_2 = distribution.mean(X)
                sigma_chol_2 = distribution.covar_chol(X, X)

                kl = func(mu_1, sigma_chol_1, mu_2, sigma_chol_2)

                return kl

        raise NotImplementedError(
            "KL of GaussianDistribution against {dist} is not implemented currently.".format(
                dist=distribution_type
            )
        )


class DiagonalNaturalGaussianDistribution(Distribution):
    """
    A Gaussian distribution parameterised by its (diagonal) Natural Parameters
    """

    def __init__(
        self,
        lambda_1: Optional[np.ndarray] = None,
        lambda_2: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        trainable: Optional[bool] = True,
    ) -> None:
        """
        Here mu, covariance_chol parameterise the natural parameters of the (implicit) likelihood

            q(u | λ₁, λ₂) = p(λ₁ | λ₂, f)p(f)/p(λ₁ | λ₂)

        """

        self.meta = meta

        if self.meta is None:
            self.meta = {}

        self.name = name
        if self.name is None:
            self.name = "DiagonalNaturalGaussianDistribution"

        super(DiagonalNaturalGaussianDistribution, self).__init__(
            self.name, self.meta, trainable
        )

        scope = "variational"

        self.lambda_1 = lambda_1
        if self.lambda_1 is not None:
            self.lambda_1 = self.parameter(
                val=self.lambda_1,
                scope=scope,
                train=trainable,
                module_name=self.name,
                param_name="mu",
            )

        self.lambda_2 = lambda_2
        if self.lambda_2 is not None:
            # TODO: check what the constraint should be
            self.lambda_2 = self.parameter(
                val=self.lambda_2,
                scope=scope,
                constraint="negative",
                train=trainable,
                module_name=self.name,
                param_name="covariance_chol",
            )

    def mean(self, X=None):
        lambda_1 = self.lambda_1.value
        lambda_2 = self.lambda_2.value

        mean, _ = natural_to_standard_diagonal(lambda_1, lambda_2)

        return mean

    def covar_diag(self, X1):
        lambda_1 = self.lambda_1.value
        lambda_2 = self.lambda_2.value

        _, var_diag = natural_to_standard_diagonal(lambda_1, lambda_2)

        return var_diag

    def covar(self, X1, X2):
        covar_diag = self.covar_diag(X1)
        return np.diag(np.squeeze(covar_diag))

    def covar_chol(self, X1, X2):
        covar_diag = self.covar_diag(X1)
        return np.diag(np.squeeze(np.sqrt(covar_diag)))


class BlockDiagonalNaturalGaussianDistribution(Distribution):
    """
    A Gaussian distribution parameterised by its (block diagonal) Natural Parameters

    Because the covariance has to be psd we parameterise the covariance directly
    """

    def __init__(
        self,
        lambda_1: Optional[np.ndarray] = None,
        covar_chol: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        trainable: Optional[bool] = True,
    ) -> None:
        """
        Here mu, covariance_chol parameterise the natural parameters of the (implicit) likelihood

            q(u | λ₁, λ₂) = p(λ₁ | λ₂, f)p(f)/p(λ₁ | λ₂)


        Let N be the number of blocks and D be the dimension of each block

        """

        self.meta = meta

        if self.meta is None:
            self.meta = {}

        self.name = name
        if self.name is None:
            self.name = "BlockDiagonalNaturalGaussianDistribution"

        super(BlockDiagonalNaturalGaussianDistribution, self).__init__(
            self.name, self.meta, trainable
        )

        scope = "variational"

        self.lambda_1 = lambda_1
        if self.lambda_1 is not None:
            # N x D x 1
            self.lambda_1 = self.parameter(
                val=self.lambda_1,
                scope=scope,
                train=trainable,
                module_name=self.name,
                param_name="mu",
            )

        self.covar_chol = covar_chol
        if self.covar_chol is not None:
            # N x D x D
            self.covar_chol = self.parameter(
                val=self.covar_chol,
                scope=scope,
                meta={"N": self.meta["N"][1]},
                constraint="block lower triangular",
                train=trainable,
                module_name=self.name,
                param_name="covariance_chol",
            )

    @property
    def lambda_2(self):
        """
            self.covar_chol is an array of blocks parameterised their its cholesky factors.\
            The natural parameter is:
                - ½S⁻¹            
            Where because S is block diagonal we only need to calculate the inverses for each of the blocks
        """
        covar_chol = self.covar_chol.value

        def block_wise_lambda_2(var_chol):
            var_chol_inverse = triangular_solve(
                var_chol, np.eye(var_chol.shape[0]), lower=True
            )
            lambda_2 = -0.5 * var_chol_inverse.T @ var_chol_inverse
            return lambda_2

        # generate the blocks of lambda_2
        lambda_2 = jax.vmap(block_wise_lambda_2, in_axes=(0), out_axes=0)(covar_chol)

        return lambda_2

    # TODO
    def mean(self, X=None):
        # N x D x 1
        lambda_1 = self.lambda_1.value
        # N x D x D
        # lambda_2 = self.lambda_2

        # calculate the natural parameters for each block
        mean = jax.vmap(
            natural_parameterised_by_covariance_to_mean, in_axes=(0, 0), out_axes=0
        )(lambda_1, self.covar(None, None))
        # mean, _ = jax.vmap(natural_to_standard, in_axes=(0, 0), out_axes=(0, 0))(lambda_1, lambda_2)

        return lambda_1

        return mean

    def covar_diag(self, X1):
        raise NotImplementedError()

    def covar(self, X1, X2):
        covar_chol = self.covar_chol.value

        def block_wise_cholesky_multiply(var_chol):
            return var_chol @ var_chol.T

        S = jax.vmap(block_wise_cholesky_multiply, in_axes=(0), out_axes=0)(covar_chol)

        return S

    def covar_chol(self, X1, X2):
        raise NotImplementedError()


class BlockGaussianDistribution(Distribution):
    """
    Constructs a joint Gaussian from a list of Gaussian distributions. Assumes independence and hence the covariance will be block diagional.
    """

    def __init__(
        self, distributions: List[GaussianDistribution], name: Optional[str] = None
    ):
        self.distributions = distributions
        if name is None:
            self.name = "BlockGaussian"
        self.name = name
        # TODO: generalise this some how
        self.X = self.distributions[0].X
        super(BlockGaussianDistribution, self).__init__(
            name=self.name, meta={}, trainable=True
        )

    def mean(self, X=None):
        mu_arr = []
        for dist in self.distributions:
            mu_arr.append(dist.mean(X))
        return np.vstack(mu_arr)

    def covar(self, X1, X2):
        blocks = []
        for dist in self.distributions:
            blocks.append(dist.covar(X1, X2))
        return jax.scipy.linalg.block_diag(*blocks)

    def covar_diag(self, X):
        arr = []
        for dist in self.distributions:
            diag = dist.covar_diag(X)
            diag = np.reshape(diag, [diag.shape[0], 1])
            arr.append(diag)
        return np.vstack(arr)


class ZeroMeanGaussianDistribution(GaussianDistribution):
    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        covariance_chol: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        trainable: Optional[bool] = True,
    ) -> None:
        if mu is not None:
            raise RuntimeError(
                "ZeroMeanGaussianDistribution: mu should not be in param list"
            )

        if name is None:
            self.name = "ZeroMeanGaussianDistribution"

        super(ZeroMeanGaussianDistribution, self).__init__(
            mu, covariance_chol, name, meta, trainable
        )

    # @jit
    def mean(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])[:, None]


class KernelGaussianDistribution(ZeroMeanGaussianDistribution):
    """
    Gaussian distribution paramterised by a kernel. Assume zero mean.
    """

    def __init__(
        self,
        kernel: Kernel,
        mu: Optional[np.ndarray] = None,
        covariance_chol: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> None:

        if name is None:
            self.name = "KernelGaussianDistribution"

        self.kernel = kernel

        self.X = meta["X"]

        super(KernelGaussianDistribution, self).__init__(
            mu, covariance_chol, name, meta, False
        )

    # @jit
    def covar(self, X1, X2):
        return self.kernel.K(X1, X2)

    # @jit
    def covar_diag(self, X1):
        return self.kernel.K_diag(X1)

    # @jit
    def covar_chol(self, X1, X2):
        v = self.covar(X1, X2)
        v = v + Settings.jitter * np.eye(v.shape[0])
        return jax.scipy.linalg.cholesky(v, lower=True)


class WhitenedKernelGaussianDistribution(KernelGaussianDistribution):
    def predict(self, X, full_covar=False):
        if full_covar:
            return self.mean(X), self.covar(X, X)

        return self.mean(X), np.diag(self.covar(X, X))

    def predict(self, XS, full_covar=False):
        k_zz = self.kernel.K(self.Z(), self.Z())
        k_xz = self.kernel.K(XS, self.Z())
        approx_posterior_mu = self.mean(None)
        approx_poster_sig = self.covar(None, None)

        if full_covar:
            k_xx = self.kernel.K(XS, XS)
            return whitened_gaussian_conditional(
                k_xx, k_xz, k_zz, approx_posterior_mu, approx_poster_sig
            )

        k_xx = self.kernel.K_diag(XS)
        return diagional_whitened_gaussian_conditional(
            k_xx, k_xz, k_zz, approx_posterior_mu, approx_poster_sig
        )


class BlockWhitenedGaussianDistribution(BlockGaussianDistribution):
    pass
