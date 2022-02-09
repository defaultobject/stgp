import objax
from .models import GP
from .kernels import RBF
from .likelihood import Gaussian, ProductLikelihood
from .transforms import Independent
from typing import List, Optional
from .sparsity import NoSparsity, FullSparsity
import warnings

def get_default_kernel(input_dim: int, num_latents: int) -> List['Kernel']:
    warnings.warn('Using default RBF kernel')
    return [
        RBF(
            lengthscales=[1.0 for d in range(input_dim)],
            input_dim=input_dim
        )
        for q in range(num_latents)
    ]

def get_default_likelihood(num_outputs: int) -> List['Likelihood']:
    warnings.warn('Using default Product Gaussian Likelihood')
    return ProductLikelihood([
        Gaussian(variance=1.0)
        for p in range(num_outputs)
    ])

def get_default_independent_prior(X, input_dim: int, num_latents: int, kernel_list: Optional['Kernel'] = None, Z: Optional['np.ndarray'] = None) -> 'Prior':
    warnings.warn('Using default Independent prior')

    if kernel_list is None:
        kernel_list = get_default_kernel(input_dim, num_latents)

    if Z is None:
        sparsity_arr = [NoSparsity(X) for q in range(num_latents)]
    elif type(Z) is list:
        sparsity_arr = [FullSparsity(Z[q]) for q in range(num_latents)]
    else:
        sparsity_arr = [FullSparsity(Z) for q in range(num_latents)]

    return Independent(
        latents = [
            GP(
                X = X,
                kernel = kernel_list[q],
                sparsity = sparsity_arr[q]
            )
            for q in range(num_latents)
        ],
        prior=True
    )
