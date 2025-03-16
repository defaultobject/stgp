""" Expectations of kernels """
import jax
import jax.numpy as np
import chex
from ..dispatch import dispatch, evoke
from ..kernels import RBF, SpatioTemporalSeperableKernel
from .gaussian import log_gaussian_scalar
from .matrix_ops import cholesky, cholesky_solve


def rbf_Li(ls, mi, si, xi):
    lam = ls **2

    const = 1 / (
        np.sqrt((si/lam)+1)
    )

    exp_term = np.exp(
        -0.5 * ((mi-xi)**2)/(lam+si)
    )

    return const*exp_term

def rbf_Lij(ls, u, s, X):
    def _rbf_Lij(ls, u, s, x_i, x_j):
        lam = ls**2

        xd = 0.5*(x_i+x_j)
        const = 1/np.sqrt(2*s/lam + 1)
        exp = np.exp(
            -0.5*(u-xd)**2/(s+lam/2)
        ) * np.exp(
            -0.5*(x_i-x_j)**2/(2*lam)
        )

        return const*exp

    res = jax.vmap(
        jax.vmap(
            lambda x_i, x_j: _rbf_Lij(ls, u, s, x_i, x_j),
            [None, 0]
        ),
        [0, None]
    )(X, X)

    # N x N
    res = np.squeeze(res)

    return res


def rbf_Mij(ls, m_i, m_j, s_i, s_j, s_ij,  x_i, x_j):
    lam = ls**2

    L = lam + s_j + s_i - 2*s_ij

    const = np.squeeze(np.sqrt(2 * np.pi*lam))

    # setup correct dimensions for log_gaussian_scalar
    k_ij = const * np.exp(log_gaussian_scalar(0, m_i-m_j, L))
    return k_ij

@dispatch(RBF)
def kernel_psi_statistics(XS, X, noise_m, noise_S, kern):
    chex.assert_rank(noise_m, 2)
    chex.assert_rank(noise_S, 2)


    first_argument_form = None
    square_form = None
    both_argument_form = None

    ls = kern.lengthscales[0]

    S_diag = np.diag(noise_S)

    sq = np.squeeze

    # [Ns x N]
    first_argument_form =  jax.vmap(
        lambda mi, si: jax.vmap(
            lambda  mi, si, xi: rbf_Li(ls, np.squeeze(mi), np.squeeze(si), np.squeeze(xi)),
            [None, None, 0]
        )(mi, si, X),
        [0, 0]
    )(noise_m, S_diag)

    # [Ns x N x N]
    L_diag =  jax.vmap(
        lambda  mi, si: rbf_Lij(ls, np.squeeze(mi), np.squeeze(si), X),
        [0, 0]
    )(noise_m, S_diag)

    # [Ns x Ns x N x N]
    L_diag = np.eye(L_diag.shape[0])[..., None, None] * L_diag

    # [N x N x Ns x Ns]
    L = jax.vmap(
        jax.vmap(
            lambda a, b: a[:, None]@b[None, :],
            [None, 1]
        ),
        [1,  None]
    )(first_argument_form, first_argument_form)

    # [Ns x Ns x N x N]
    L = np.transpose(L, [2, 3, 0, 1])
    # [N x N x Ns]
    L_first_diag = np.diagonal(L, axis1=0, axis2=1)
    L_first_diag = np.eye(L_first_diag.shape[-1])[None, None, ...] * L_first_diag[..., None]

    # [Ns x Ns x N x N]
    L_first_diag = np.transpose(L_first_diag, [2, 3, 0, 1])
     
    # [Ns x Ns x N x N]
    square_form = L - L_first_diag + L_diag

    # [Ns x Ns]
    both_argument_form = jax.vmap(
        jax.vmap(lambda x_i, x_j, m_i, m_j, s_i, s_j, s_ij: rbf_Mij(ls, sq(m_i), sq(m_j), sq(s_i), sq(s_j), sq(s_ij), x_i, x_j), [0, None, 0, None, 0, None, 0]),
        [None, 0, None, 0, None, 0, 0]
    )(XS, XS, noise_m, noise_m , S_diag, S_diag, noise_S)


    return first_argument_form, square_form, both_argument_form

@dispatch(SpatioTemporalSeperableKernel)
def kernel_psi_statistics(XS, X, noise_m, noise_S, kern):
    breakpoint()


def get_psi_statistics_linear_form_from_mu_var_from_gram(
    XS, X, Y, K_ss, K_sx, K_xx, likelihood_var, noise_pred_mu, noise_pred_var, base_gp_kernel
):
    N = X.shape[0]

    first_argument_form, square_form, both_argument_form = evoke('kernel_psi_statistics', base_gp_kernel)(
        XS, X, noise_pred_mu, noise_pred_var, base_gp_kernel
    )

    K_tilde = K_xx + likelihood_var
    K_tilde_chol = cholesky(K_tilde)

    # TODO: can we avoid direct solve?
    K_tilde_inv = cholesky_solve(K_tilde_chol, np.eye(N))
    beta = cholesky_solve(K_tilde_chol, Y)

    mean_weights = (cholesky_solve(K_tilde_chol, first_argument_form.T).T)

    var = both_argument_form
    var = var - np.sum(np.multiply((K_tilde_inv - beta @ beta.T)[None, None, ...], square_form), axis=[2, 3])

    llT = jax.vmap(
        jax.vmap(
            lambda a, b: a[:, None]@b[None, :],
            [None, 1]
        ),
        [1,  None]
    )(first_argument_form, first_argument_form)

    # TODO: should probably write as a scan to save memory!
    var = var - np.sum(np.multiply( np.transpose(llT, [2, 3, 0, 1]), (beta @ beta.T)[None, None, ...]), axis=[2, 3])

    return mean_weights, var

def get_psi_statistics_linear_form_from_mu_var(noise_pred_mu, noise_pred_var, XS, data, base_gp, likelihood, prior, block_size):
    """
    Only supported for a single latent GP. 

    args:
        noise_pred_mu: 1 x N x 1 
        noise_pred_var: 1 x 1 x N x N
    """

    chex.assert_rank([noise_pred_mu, noise_pred_var], [3, 4])

    # remove the latent function dimension
    noise_pred_mu = noise_pred_mu[0]
    noise_pred_var = noise_pred_var[0][0]

    return get_psi_statistics_linear_form_from_mu_var_from_gram(
        XS, 
        data.X,
        data.Y,
        K_ss = base_gp.covar(XS, XS),
        K_sx = base_gp.covar(XS, data.X),
        K_xx = base_gp.covar(data.X, data.X),
        likelihood_var = likelihood.variance * np.eye(data.N),
        noise_pred_mu = noise_pred_mu, 
        noise_pred_var = noise_pred_var,
        base_gp_kernel = base_gp.kernel
    )


def get_psi_statistics_linear_form(XS, data, gp, likelihood, prior, block_size):
    base_gp = prior.base_prior
    noise_gp = prior.prediction_gp

    noise_pred_mu, noise_pred_var = noise_gp.predict_f(XS, diagonal=False, fix_shapes = False)

    # TODO: shapes are all wrong :) 
    noise_pred_mu = np.transpose(noise_pred_mu, [0, 2, 1])

    #noise_pred_mu = np.array([XS])
    #noise_pred_var = noise_pred_var*0.0

    return get_psi_statistics_linear_form_from_mu_var(noise_pred_mu, noise_pred_var, XS, data, base_gp, likelihood, prior, block_size)

def get_fitc_sparsity_transformation():
    pass
