from ..module import Module
from ..dispatcher import Dispatcher
from ..data import Data, ListData, PlaceholderData
from ..sparsity import Sparsity
from ..distributions import *
from ..likelihoods import Likelihood 


import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial

import numpy as onp

import typing
from typing import List, Callable, Optional, Tuple
from numpy.polynomial.hermite import hermgauss
import itertools

#TODO - NEED TO CLEAN

# Todo is there a nicer way than this? Can i just pass the callable as a lambda and treat as a static argument? or will this mess up gradients?

# Below mean_field_gaussian_monte_carlo requires a callable function. To make all arguments jit'able we provide some
#   standard use cases 

class IntegralCallable(Module):
    """

    """
    def __init__(self, name:Optional[str]='integral_callable'):
        self.name = name
        super(IntegralCallable, self).__init__(name=name)

    def forward(self):
        pass


def forward(samples, data, likelihood):
    return np.sum(likelihood.log_likelihood(data, samples))

def get_distributions(XS, data, model, distribution_arr):
    #get mean and variances of the Gaussian distributions
    mu_arr, var_arr = [], []

    for q, component in enumerate(distribution_arr):

        if type(XS) is list:
            #TODO: need to define a standard how to handle the X/Z that the latents are defined on
            Z = XS[0]
        else:
            Z = XS

        mu_q, var_diag_q = distribution_arr[q].predict_f(Z, data, model, q, diagonal_var=True, predict=True)

        mu_q = np.reshape(mu_q, [mu_q.shape[0], 1])
        var_diag_q = np.reshape(var_diag_q, [var_diag_q.shape[0], 1])
        mu_arr.append(mu_q)
        var_arr.append(var_diag_q)

    return mu_arr, var_arr

class ELLCallable(IntegralCallable):
    """

    """
    def __init__(self, name:Optional[str]='ell_callable'):
        self.name = name

        super(ELLCallable, self).__init__(name=name)

    def forward(self, samples, data, model):
        return np.sum(model.likelihood.log_likelihood(data, samples))

    def get_distributions(self, XS, data, model, distribution_arr):
        #get mean and variances of the Gaussian distributions
        mu_arr, var_arr = [], []

        for q, component in enumerate(distribution_arr):

            if type(XS) is list:
                #TODO: need to define a standard how to handle the X/Z that the latents are defined on
                Z = XS[0]
            else:
                Z = XS

            mu_q, var_diag_q = distribution_arr[q].predict_f(Z, data, model, q, diagonal_var=True, predict=True)

            mu_q = np.reshape(mu_q, [mu_q.shape[0], 1])
            var_diag_q = np.reshape(var_diag_q, [var_diag_q.shape[0], 1])
            mu_arr.append(mu_q)
            var_arr.append(var_diag_q)

        return mu_arr, var_arr

class IdentityCallable(ELLCallable):
    """

    """
    def __init__(
            self, 
            distribution_arr: List[Distribution], 
            prior_arr:  List[Distribution], 
            sparsity_arr: List[Sparsity],
            name:Optional[str]='identity_callable'
    ):

        super(IdentityCallable, self).__init__(None, None, distribution_arr, prior_arr, sparsity_arr, name)
   
    def forward(self, samples):
        
        return samples

class IdentityPrecomputedCallable(ELLCallable):
    """

    """
    def __init__(
            self, 
            distribution_arr: List[Distribution], 
            prior_arr:  List[Distribution], 
            sparsity_arr: List[Sparsity],
            mu_arr: List[np.ndarray],
            var_arr: List[np.ndarray],
            name:Optional[str]='IdentityPrecomputedCallable'
    ):

        self.mu_arr = mu_arr
        self.var_arr = var_arr

        super(IdentityPrecomputedCallable, self).__init__(None, None, distribution_arr, prior_arr, sparsity_arr, name)


    def get_distributions(self, XS):
        return self.mu_arr, self.var_arr

   
    def forward(self, samples):
        
        return samples


@partial(jit, static_argnums=(4))  
def mean_field_gaussian_samples(key, data: Data, likelihood: 'Likelihood', distribution_arr: List[Distribution], num_samples:int, mu_arr, var_arr):
    #create key to generate samples with and key to return

    rng_key, new_key = jax.random.split(key)

    XS = data.X[0]

    #create num_samples keys
    rng_keys = jax.random.split(rng_key, num_samples)  # (num_samples,)

    #get the mean and var of p(a_1), ..., p(a_p)
    #mu_arr, var_arr = get_distributions(XS, data, model , distribution_arr )

    num_latents = len(mu_arr)

    #for each random key we create a sample and evaluate f(a^{(s)}_1, ..., a^{(s)}_p)
    @jit
    def bb(key):
        #create a random key per latent distribution
        keys = jax.random.split(key, num_latents)

        #for each sample use the reparamterisation trick:
        # 1) s_p  ∼ N(0, 1)
        # 2) a^{(s)}_p = mean_{a_p} + std_{a_p} * s_p
        samples = []
        for q in range(num_latents):
            normal_sample = jax.random.normal(keys[q], shape=mu_arr[q].shape) 

            f_sample = mu_arr[q] + np.sqrt(var_arr[q])*normal_sample

            samples.append(f_sample)

        #call `f` with a single sample from all num_latent distributions
        return forward(samples, data, likelihood)

    #bb is defined for a single sample. use vmap to batch bb for all samples.
    run_bb = jax.vmap(bb, in_axes=(0), out_axes=0)

    #get result over all samples
    monte_carlo_results = run_bb(rng_keys)

    return new_key, monte_carlo_results

@partial(jit, static_argnums=(1))  
def test(model: 'Model', num_samples:int):
    model.key = model.key
    return np.sum(model.sparsity.Z)

@partial(jit, static_argnums=(2))  
def hhh_mean_field_gaussian_monte_carlo(key, model: 'Model', num_samples:int):
    """
        See https://rlouf.github.io/post/jax-random-walk-metropolis/ 

        Args:
            key:
            XS:
            f: 
            distribution_arr:
            sparsity_arr:

        Method:

            This method approximates expectations of the form:

                ∫ f(a_1, ..., a_p) p(a_1)...p(a_p) da_1 ... da_p
            
            where p(a_1), ..., p(a_p) are all Gaussians defined by distribution_arr with sparsity defined by sparsity_arr.

            Each of the Gaussians are independent and so the expectation can be approximated by independently
                sampling from the distributions :

                ≈ (1/S) ∑ f(a^{(s)}_1, ..., a^{(s)}_p) where a^{(s)}_1, ..., a^{(s)}_p ∼ p(a_1)...p(a_p)

        Returns:
            key, estimate
    """

    
    return key, np.sum(model.sparsity.Z)


    
    if False:
        XS = model.data.X[0]
        mu_arr, var_arr = get_distributions(XS, model.data, model ,  model.inference.variational_posterior.components )
        #return key, np.sum(model.sparsity.Z)
        #return key, np.sum(model.inference.variational_posterior.components[0].distribution.mean())
        new_key, monte_carlo_results = mean_field_gaussian_samples(key, model.data, model.likelihood, model.inference.variational_posterior.components, num_samples, mu_arr, var_arr)

        res = np.mean(monte_carlo_results, axis=0)

        return new_key, res

def gauss_hermite(H):
    return  hermgauss(H)

def mvhermgauss(H: int, D: int):
    """
    This function is taken from GPflow: https://github.com/GPflow/GPflow
    Copied here rather than imported so that users don't need to install gpflow to use kalman-jax
    LICENSE:
        Copyright The Contributors to the GPflow Project. All Rights Reserved.
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.
    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])
    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w

def mean_field_gauss_hermite_quadrature(data, likelihood, model, num_quad_points=10):

    #x in S**D x D
    #w in S**D 
    num_latents = model.total_num_latents
    x, w = mvhermgauss(num_quad_points, model.total_num_latents)
    const = np.pi**-0.5

    XS = data.X[0]

    q = model.inference.variational_posterior
    mu_arr, var_arr = q.predict_f(XS, data, model, diagonal_var=True)

    mu = np.hstack(mu_arr) #NxD
    var = np.hstack(var_arr) #NxD
    sig = np.sqrt(var) #NxD

    def batched_log_likelihood(data, x_s, mu, sig):
        y = 2.0**0.5 * sig*x_s + mu
        ll = likelihood.log_likelihood(data, [y[:, q][:, None] for q in range(num_latents)])
        return np.squeeze(ll)

    res = jax.vmap(batched_log_likelihood, (None, 0, None, None), (0))(data, x, mu, sig)
    return np.sum(w * const*res)


#@partial(jit, static_argnums=(2))  
def gauss_hermite_quadrature(data, likelihood, model, num_quad_points=10):



    x, w = gauss_hermite(num_quad_points)
    const = np.pi**-0.5

    XS = data.X[0]

    #mu_arr, var_arr = get_distributions(XS, data, model,  model.inference.variational_posterior.components)
    q = model.inference.variational_posterior
    mu_arr, var_arr = q.predict_f(XS, data, model, diagonal_var=True, predict=False)

    num_latents = len(mu_arr)

    for q in range(num_latents):
        mu_q, var_q = mu_arr[q], var_arr[q]

        sig_q = np.sqrt(var_q)

        #change of variable
        y = 2.0**0.5*sig_q*x + mu_q   

        def batched_log_likelihood(data, f):
            #f = np.expand_dims(f, -1)
            ll =  likelihood.log_likelihood(data, [f])
            return ll

        #N x S x 1
        res = jax.vmap(batched_log_likelihood, (None, 1), (1))(data, y)

        print('res: ', res.shape)

        return np.sum(w * const*res[..., 0])

def precomputed_mean_field_gauss_hermite_quadrature(data, likelihood, mu_arr, var_arr, num_quad_points=10):

    x, w = gauss_hermite(num_quad_points)
    const = np.pi**-0.5

    XS = data.X[0]

    num_latents = len(mu_arr)

    for q in range(num_latents):
        mu_q, var_q = mu_arr[q], var_arr[q]

        mu_q = np.reshape(mu_q, [-1, 1])
        var_q = np.reshape(var_q, [-1, 1])

        sig_q = np.sqrt(var_q)

        #change of variable
        y = 2.0**0.5*sig_q*x + mu_q   

        def batched_log_likelihood(data, f):
            f = np.expand_dims(f, -1)
            lik = likelihood.log_likelihood(data, [f])
            return lik

        res = jax.vmap(batched_log_likelihood, (None, 1), (1))(data, y)

        return np.sum(w * const*res[..., 0])
        
#@partial(jit, static_argnums=(3))  
def predict_gauss_hermite_quadrature(data_xs, data, likelihood, model, num_quad_points=10):

    x, w = gauss_hermite(num_quad_points)
    const = np.pi**-0.5

    XS = data_xs.X

    mu_arr, var_arr = get_distributions(XS, data, model,  model.inference.variational_posterior.components)

    num_latents = len(mu_arr)

    for q in range(num_latents):
        mu_q, var_q = mu_arr[q], var_arr[q]

        sig_q = np.sqrt(var_q)

        #change of variable
        y = 2.0**0.5*sig_q*x + mu_q   

        def batched_log_likelihood(f):
            print('f: ', f.shape)
            f = np.expand_dims(f, -1)
            f = likelihood.eval(f)
            return f

        first = jax.vmap(batched_log_likelihood, (1), (1))(y)
        second = first**2

        first = np.sum(w * const*first[..., 0], axis=1)
        second = np.sum(w * const*second[..., 0], axis=1)

        first = np.expand_dims(first, -1)
        second = np.expand_dims(second, -1)

        return first, second - first**2

def predict_mean_field_gauss_hermite_quadrature(data_xs, data, likelihood, model, num_quad_points=10):
    """
        Let
            S = the number quadrature points
            D = the number of latent functions
            N = the number of prediction points
            P = number of outputs

        This methods approximates the first two moments :
            p(Y | Xs) = \prod  p(Y_p | Xs)

            p(Y_p | Xs) = \int  p(Y_p | [f_1, ..., f_Q], Xs) q(f_1) ... q(f_Q) df_1 ... f_Q

        with quadrature

        The first moment is:
            
            m1 = \int  \int Y_p p(Y_p | [f_1, ..., f_Q], Xs) dY_p q(f_1) ... q(f_Q) df_1 ... f_Q 
               =  \int .. \int  lik_mean  q(f_1) ... q(f_Q) df_1 ... f_Q
               \approx 
    
    """

    num_latents = model.total_num_latents
    num_outputs = model.num_outputs

    #x in S**D x D
    #w in S**D 
    x, w = mvhermgauss(num_quad_points, num_latents)

    print('x: ', x.shape)
    print('w: ', w.shape)
    print('num_latents: ', num_latents)

    const = np.pi**(-0.5*num_latents)

    q = model.inference.variational_posterior
    #TODO: generalise to multiple latnt functions?
    #TODO: generalise with dispatch
    try:
        mu_arr, var_arr = q.predict_f(data_xs, data, model, latent=0, diagonal_var=True, spatial_predict=True, temporal_predict=True)
        grid_sort_idx = data_xs.order_indexes['grid_sort_idx']

        mu_arr = mu_arr[grid_sort_idx, :]
        var_arr = var_arr[grid_sort_idx, :]


        if type(mu_arr) is not list:
            mu_arr = [mu_arr]
            var_arr = [var_arr]

    except Exception as e:
        print('exception: ', e)
        mu_arr, var_arr = q.predict_f(data_xs, data, model, diagonal_var=True, predict=True)

    mu = np.hstack(mu_arr) #NxD
    var = np.hstack(var_arr) #NxD
    sig = np.sqrt(var) #NxD

    #Batch over each of the S**D samples 
    def batched_likelihood(x_s, mu, sig):

        #Batch over the data
        def batched_data(x_s, mu, sig):
            """
                x_s in D
                mu in D
                sig in D
            """
            #for each datapoint compute the likelihood

            y = 2.0**0.5 * sig*x_s + mu # D
            y = y[:, None, None] #  D x 1 x 1
            ll_1 = likelihood.conditional_mean([y[q] for q in range(num_latents)])
            ll_1 =  np.hstack(ll_1)
            return ll_1.reshape([-1, num_outputs])
    
        #batch over data
        #N x D x P
        d =  jax.vmap(batched_data, (None, 0, 0), (0))(x_s, mu, sig)
        return d

    lik_var = likelihood.conditional_var(mu)

    #batch over samples
    first= jax.vmap(batched_likelihood, (0, None, None), (0))(x, mu, sig) #S x Ns x P

    print('first: ', first.shape)

    second = first**2 #S x Ns x P

    mean_arr =[]
    var_arr = []
    for p in range(num_outputs):
        first_p = first[:, :, p] # S x Ns
        second_p = second[:, :, p] # S x Ns

        def batched_over_data(w, const, moment):
            """
                Args:
                    w in S^3
                    moment in S^3
            """
            return np.sum(np.squeeze(w)*const*np.squeeze(moment))

        first_p = jax.vmap(batched_over_data, (None, None, 1), (0))(w, const, first_p) #S x Ns x P
        second_p = jax.vmap(batched_over_data, (None, None, 1), (0))(w, const, second_p) #S x Ns x P

        first_p = np.expand_dims(first_p, -1)
        second_p = np.expand_dims(second_p, -1)

        mean_p = first_p
        var_p = lik_var[p] + second_p - first_p**2

        mean_arr.append(mean_p)
        var_arr.append(var_p)
    
    return mean_arr, var_arr


