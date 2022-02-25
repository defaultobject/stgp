import jax
import jax.numpy as np
import objax
import chex

def gauss_quad():
    pass

def mv_gauss_quad():
    pass

def monte_carlo():
    pass

def mv_indepentdent_monte_carlo(fn, mu_arr, var_arr, fn_args =[], generator=None, num_samples=100, average=True):
    """
    multi-variate monte-carlo 
    """
    chex.assert_equal(mu_arr.shape, var_arr.shape)

    white_samples = objax.random.normal([num_samples]+list(mu_arr.shape), mean=0.0, stddev=1.0, generator=generator)

    # Reparameterise
    def reparameterise(fn, samples, mu_arr, var_arr, *args):
        chex.assert_equal(samples.shape, mu_arr.shape)

        s = mu_arr + samples * np.sqrt(var_arr)
        return fn(s, *args)

    num_args = len(fn_args)

    # batch over samples
    fn_samples = jax.vmap(
        reparameterise,
        [None, 0, None, None] + [None]*num_args,
        0
    )(fn, white_samples, mu_arr, var_arr, *fn_args)

    if average:
        # Return average across all samples
        return np.mean(fn_samples, axis=0)
    else: 
        return fn_samples


def mv_block_monte_carlo(fn, mu_arr, var_arr, fn_args =[], generator=None, num_samples=100, average=True):
    chex.assert_equal(mu_arr.shape[0], var_arr.shape[0])
    chex.assert_equal(mu_arr.shape[1], var_arr.shape[1])
    chex.assert_equal(mu_arr.shape[1], var_arr.shape[2])

    white_samples = objax.random.normal([num_samples]+list(mu_arr.shape), mean=0.0, stddev=1.0, generator=generator)

    chol_arr = np.linalg.cholesky(var_arr)

   # Reparameterise
    def reparameterise(fn, samples, mu_arr, chol_arr, *args):
        chex.assert_equal(samples.shape, mu_arr.shape)

        samples = samples[..., None]
        mu_arr = mu_arr[..., None]

        s = mu_arr + chol_arr @ samples 

        s = s[..., 0].T

        return fn(s, *args)


    num_args = len(fn_args)

    # batch over samples
    fn_samples = jax.vmap(
        reparameterise,
        [None, 0, None, None] + [None]*num_args,
        0
    )(fn, white_samples, mu_arr, chol_arr, *fn_args)

    if average:
        # Return average across all samples
        return np.mean(fn_samples, axis=0)
    else: 
        return fn_samples


