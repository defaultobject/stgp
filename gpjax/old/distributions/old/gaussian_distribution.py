from . import Distribution
from ..parameter import Parameter
from ..likelihoods import *
from ..computation import *
from ..computation.general import log_chol_matrix_det, cholesky_solve
from ..settings import Settings

import jax
import jax.numpy as np
from jax import jit, partial

from ..decorators import return_gradients

from jax.config import config

config.update("jax_enable_x64", True)


class GaussianDistribution(Distribution):
    def __init__(
        self,
        params: dict = None,
        name: str = None,
        meta: dict = None,
        trainable: bool = True,
    ) -> None:
        self.meta = meta

        if self.meta is None:
            self.meta = {}

        self.name = name
        if self.name is None:
            self.name = "GaussianDistribution"

        self.params = params
        if self.params is None:
            self.params = self.get_default_params()

        super(GaussianDistribution, self).__init__(
            self.params, self.name, self.meta, trainable
        )

        self.key = jax.random.PRNGKey(0)

        scope = "variational"
        if "mean" in self.params.keys():
            self.mean_param = self.parameter(
                val=self.params["mean"],
                scope=scope,
                train=trainable,
                module_name=self.name,
                param_name="mean",
            )

        if "covariance_chol" in self.params.keys():
            self.covar_sqrt_param = self.parameter(
                val=self.params["covariance_chol"],
                meta={"N": self.meta["N"]},
                constraint="lower triangular",
                train=trainable,
                module_name=self.name,
                param_name="covariance_chol",
            )

        self.kl_dict = {GaussianDistribution: gaussian_kl}

    def get_default_params(self):
        self.meta["N"] = 1

        return {"mean": np.array(1.0), "covariance_chol": np.array(1.0)}

    def mean(self, X=None):
        return self.mean_param.value

    @jit
    def covar(self, X1, X2):
        covar_sqrt = self.covar_sqrt(X1, X2)
        return np.matmul(covar_sqrt, covar_sqrt.T)

    def covar_sqrt(self, X1, X2):
        return self.covar_sqrt_param.value

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
                sigma_chol_1 = self.covar_sqrt(X, X)

                mu_2 = distribution.mean(X)
                sigma_chol_2 = distribution.covar_sqrt(X, X)

                kl = func(mu_1, sigma_chol_1, mu_2, sigma_chol_2)

                return kl

        raise NotImplementedError(
            "KL of GaussianDistribution against {dist} is not implemented currently.".format(
                dist=distribution_type
            )
        )


class SparseGaussianDistribution(GaussianDistribution):
    def __init__(
        self,
        params: dict = None,
        name: str = None,
        meta: dict = None,
        trainable: bool = True,
    ) -> None:

        if name is None:
            self.name = "SparseGaussianDistribution"

        if "Z" not in params.keys():
            raise RuntimeError("{name}: Z should be in param list".format(name=name))

        if "kernel" not in meta.keys():
            raise RuntimeError(
                "{name}: kernel should be in param list".format(name=name)
            )

        GaussianDistribution.__init__(self, params, name, meta, trainable)

        self.Z = self.parameter(
            val=self.params["Z"],
            scope="variational",
            train=trainable,
            module_name=self.name,
            param_name="Z",
        )
        self.kernel = meta["kernel"]

    def predict(self, XS, full_covar=False):
        if full_covar:
            return self.mean(X), self.covar(X, X)

        k_zz = self.kernel.K(self.Z(), self.Z())
        k_xx = self.kernel.K(XS, XS)
        k_xz = self.kernel.K(XS, self.Z())
        approx_posterior_mu = self.mean(None)
        approx_poster_sig = self.covar(None, None)

        mu, sig = gaussian_conditional(
            k_xx, k_xz, k_zz, approx_posterior_mu, approx_poster_sig
        )

        return mu, np.diag(sig)


class WhitenedGaussianDistribution(SparseGaussianDistribution):
    @return_gradients
    def KL(self, distribution=None, X=None):
        if distribution is not None:
            # raise RuntimeError('KL: distribution must not be specified')
            pass

        mu_1 = self.mean(X)
        sigma_chol_1 = self.covar_sqrt(X, X)
        kl = whitened_gaussian_kl(mu_1, sigma_chol_1)

        return kl

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

    def sample_predict(self, XS, sample):
        mu, sig = self.predict(XS, full_covar=True)
        key = jax.random.PRNGKey(sample)
        return jax.random.multivariate_normal(
            key, mu[:, 0], sig + Settings.jitter * np.eye(mu.shape[0])
        )


class ZeroMeanGaussianDistribution(GaussianDistribution):
    def __init__(
        self,
        params: dict = None,
        name: str = None,
        meta: dict = None,
        trainable: bool = True,
    ) -> None:
        if "mean" in params.keys():
            raise RuntimeError(
                "ZeroMeanGaussianDistribution: mean should not be in param list"
            )

        if name is None:
            self.name = "ZeroMeanGaussianDistribution"

        super(ZeroMeanGaussianDistribution, self).__init__(
            params, name, meta, trainable
        )

    @jit
    def mean(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])[:, None]


class KernelGaussianDistribution(ZeroMeanGaussianDistribution):
    """
    Gaussian distribution paramterised by a kernel. Assume zero mean.
    """

    def __init__(
        self,
        params: dict = None,
        name: str = None,
        meta: dict = None,
        trainable: bool = True,
    ) -> None:
        if params is None:
            params = {}

        if name is None:
            self.name = "KernelGaussianDistribution"

        if "kernel" not in meta.keys():
            raise RuntimeError(
                "KernelGaussianDistribution: Kernel should be in param list"
            )

        self.kernel = meta["kernel"]

        super(KernelGaussianDistribution, self).__init__(params, name, meta, trainable)

    @jit
    def covar(self, X1, X2):
        return self.kernel.K(X1, X2)

    @jit
    def covar_sqrt(self, X1, X2):
        v = self.covar(X1, X2)
        v = v + Settings.jitter * np.eye(v.shape[0])
        return jax.scipy.linalg.cholesky(v, lower=True)


class DiagionalGaussianDistribution(GaussianDistribution):
    """
    Gaussian distribution paramterised by a diagional covariance matrix.
    """

    def __init__(self, init=None, shapes=None, name=None, train=True):
        GaussianDistribution.__init__(
            self, init=init, shapes=shapes, name=name, train=train
        )

        if name is None:
            self.name = "DiagionalGaussianDistribution"

        if "diag" not in init.keys():
            raise RuntimeError(
                "DiagionalGaussianDistribution: Kernel should be in param list"
            )

        self.covar_diag = Parameter(
            init=init["diag"], train=train, name=self.name + "/CovarDiag"
        )

    def covar(self, X1=None, X2=None):
        print("DiagionalGaussianDistribution: ", self.covar_diag.val)
        return np.diag(self.covar_diag.val)

    def covar_sqrt(self, X1=None, X2=None):
        v = self.covar(X1, X2)
        # return jax.scipy.linalg.cholesky(v, lower=True)
        return jax.scipy.linalg.cholesky(v, lower=True)
