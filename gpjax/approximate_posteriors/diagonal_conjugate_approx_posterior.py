from . import ApproximatePosterior
from . import GaussianApproxPosterior

#from ..models import Model
from .. import Likelihood
from .. import Distribution

from ..data import Data, ListData, PlaceholderData, TimeseriesData
from ..sparsity import NoSparsity

from ..likelihoods import LMC_Likelihood, DiagonalGaussianLikelihood, NaturalDiagonalGaussianLikelihood
from ..distributions import GaussianDistribution, DiagonalNaturalGaussianDistribution
from ..settings import Settings

from ..computation.conditionals import *
from ..computation.kullback_leiblers import *
from ..computation.expected_log_likelihoods import *
#from ..computation.predictors import *
from ..computation.expectation_approximators import *

from ..computation.general import cholesky, cholesky_solve, inv_positive_transform

from ..decorators import *

import jax
import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing
from typing import Optional, List, Union, Tuple

import inspect

import warnings

class DiagonalConjugateApproxPosterior(GaussianApproxPosterior):
    def __init__(self, lambda_1:Optional[np.ndarray]=None, covariance:Optional[np.ndarray]=None, distribution: Optional[Distribution] = None, key:Optional[jax.random.PRNGKey]=None, name:Optional[str]=None,  batch_inference=None, conjugate_model = None,  data=None, sparsity=None, kernel=None, options={}):
        """
            Args:
                m: np.ndarray inital mean values for q(.)
                S: np.ndarray inital variance values for q(.)
        """

        #save __init__ arguments as properties of this object
        self.save_inputs_to_properties(locals())
        dim = self.data.X_onp[0].shape[0]
        self.dim = dim

        super(DiagonalConjugateApproxPosterior, self).__init__(dim=dim, name=self.name)

        if lambda_1 is None:
            #close to zero
            lambda_1 =  1e-5*onp.ones(dim)[:, None] #Nx1

            lambda_1 = 1e-5*np.ones([dim, 1])
            covariance = inv_positive_transform(np.ones([dim, 1]))

        if distribution is None:
            self.distribution = NaturalDiagonalGaussianLikelihood(
                lambda_1=lambda_1,
                covariance=covariance,
                meta = {
                    'N': self.dim
                }
            )

        else:
            self.distribution = distribution

        if inspect.isclass(self.conjugate_model):
            self.conjugate_model = self.conjugate_model(
                X=self.data.X, 
                Y=self.data.Y, #this will not be jitted properly and is passed as a dummy, Must pass through sep.
                inference=self.batch_inference, 
                likelihood=self.distribution,
                options=self.options, 
                set_defaults=False, 
                sparsity=self.sparsity,
                kernel=self.kernel
            )
      
    def setup(self) -> None:
        pass

    @property
    def approx_data(self):
        if type(self.sparsity) is NoSparsity:
            return TimeseriesData([self.data.X[0]], [self.distribution.Y], [self.data.X_onp[0]], None)
        else:
            raise NotImplementedError()
            #return TimeseriesData([self.sparsity.Z], [self.distribution.Y], [self.data.X_onp[0]], None)


    @ensure_data_passed
    def predict_f(self, data_xs: Data, data: Data, model: 'Model', latent: int, diagonal_var, predict=False) -> Tuple[np.ndarray, np.ndarray]:

        #when predicting with conjugate model with use the approximate data
        data = self.approx_data

        if predict:
            mean, var = self.conjugate_model.predict_y(data_xs, data, diagonal_var=True)
        else:
            mean, var = self.conjugate_model.posterior(data, diagonal_var=True)

        mean, var = mean[latent], var[latent] 

        return mean, var

    def get_approx_marginal_likelihood(self,  model: 'Model'):
        return -self.conjugate_model.get_objective(data=self.approx_data, return_grad=False, jit=False)
