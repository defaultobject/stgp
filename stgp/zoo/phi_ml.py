import jax
import jax.numpy as np

from ..transforms.multi_output import LMC
from .sde_diff import diff_cvi_sde_vgp

def helmholtz(
    X, 
    Y,
    time_kernel = None,
    space_kernel = None,
    space_diff_kernel = None,
    hierarchical = False,
    Zs = None,
    lik_var = 1.0,
    fix_y = False,
    meanfield = False,
    verbose=True,
    keep_dims=None,
    model = None,
    parallel=False
):
    """
        Args:
            model: [batch, sde_cvi]
    """

    if model is None:
        model = 'sde_cvi'


    # only defined for 2d methods atm
    assert X.shape[1] == 2

    def prior_fn(latents):
        """
        The Helmholtz prior is defined as 
            [df1/dx df1/dy]^T + [-df2/dy df2/dx]^T
        """
        #Holmholtz prior

        W = np.array([
            # [f, fs, ft, fts]_q
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, -1, 0, 0],
        ])
        out_dim = 2
        in_dim = 8

        lmc_prior =  LMC(
            latents=latents,
            input_dim=in_dim,
            output_dim=out_dim,
            W = W
        )

        lmc_prior._W.fix()

        return lmc_prior

    if model == 'sde_cvi':
        return diff_cvi_sde_vgp(
            X, 
            Y,
            num_latents=2,
            time_diff = 1,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            space_diff_kernel = space_diff_kernel,
            hierarchical = hierarchical,
            Zs = Zs,
            lik_var = lik_var,
            fix_y = fix_y,
            meanfield = meanfield,
            prior_fn = prior_fn,
            verbose=verbose,
            keep_dims=keep_dims,
            parallel=parallel
        ) 
    else:
        raise NotImplementedError()


