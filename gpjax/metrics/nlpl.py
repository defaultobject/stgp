from ..computation.gaussian import log_gaussian_diagonal 
import numpy as onp

def gaussian_negative_log_predictive_likelihood(Y, pred_mu, pred_var):
    return float(onp.squeeze(onp.array(log_gaussian_diagonal(Y, pred_mu, pred_var))))

