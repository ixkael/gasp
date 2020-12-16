# -*- coding: utf-8 -*-

import numpy as np

# This is a scale-free likelihood function, marginalizing over the scale of the magnitudes
def scalefree_flux_marginalised_likelihood(
    f_obs, f_obs_invvar, f_mod, magerror_floor=0
):
    """
    Scale-free likelihood: given a model f_mod and some data f_obs,
    multidimensional (for example, N_bands photometry),
    computes the best-fit multiplicative scaling and the (analytic) Bayesian evidence
    with the scaling parameter marginalised over.
    Applicable to many objects and a grid of model at once, provided the shapes
    can be broadcasted.
    The prior on the scaling parameter is uniform, but the equations can be easily modified
    to include a Gaussian prior, since the Bayesian evidence is still analytic.

    Parameters
    ----------
    f_obs, f_obs_invvar: ndarray (..., N_bands)
        Fluxes and flux inverse variances.
        Non-finite values are ignored in calculations.
        Typical shape is (N_obj, 1, N_bands)
    f_mod: ndarray (..., N_bands)
        Flux model to be scaled. Shape should be compatible with f_obs,
        such that f_mod * f_obs would work.
        Typical shape is (1, N_templates, N_bands)
    magerror_floor:
        Minimum magnitude error. Flux errors will be modified accordingly.

    Returns
    -------
    loglike: ndarray (..., )
        Marginalised likelihood (or evidence) with scaling parameter marginalised.
        Typical shape is (N_obj, N_templates)
    ellML: ndarray (..., )
        Maximum likelood value of the scaling, i.e. maximising the chi2 between f_obs and f_mod,
        specifically f_obs_invvar * (f_obs - f_mod * ellML[..., None])**2
        Typical shape is (N_obj, N_templates)

    """
    ind = 1 / (f_obs * f_obs_invvar) < magerror_floor
    if np.sum(ind) > 0:
        f_obs_invvar[ind] = 1 / (magerror_floor * f_obs[ind])
    # ind = f_obs * f_obs_invvar ** 0.5 < 1e-6
    ind = ~np.isfinite(f_obs)
    ind |= ~np.isfinite(f_obs_invvar)
    log_var = np.where(ind, 0, -np.log(f_obs_invvar))
    invvar = np.where(ind, 0.0, f_obs_invvar)
    FOT = np.sum(f_mod * f_obs * invvar, axis=-1)
    FTT = np.sum(f_mod ** 2 * invvar, axis=-1)
    FOO = np.sum(f_obs ** 2 * invvar, axis=-1)
    logSigma_det = np.sum(log_var, axis=-1)  # (nobj, ..., )
    ellML = FOT / FTT
    chi2 = FOO - (FOT / FTT) * FOT
    loglike = -0.5 * (chi2 + np.log(FTT) + logSigma_det)
    ellML = FOT / FTT
    return loglike, ellML
