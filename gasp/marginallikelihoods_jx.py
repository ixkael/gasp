# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import jit, vmap


def logmarglike_lineargaussianmodel_onetransfer(M_T, y, yinvvar, logyinvvar=None):
    """
    Fit linear model to one Gaussian data set, with no (=uniform) prior on the linear components.

    Parameters
    ----------
    y, yinvvar, logyinvvar : ndarray (n_pix_y)
        data and data inverse variances.
        Zeros will be ignored.
    M_T : ndarray (n_components, n_pix_y)
        design matrix of linear model

    Returns
    -------
    logfml : ndarray scalar
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (n_components)
        Best fit MAP parameters
    theta_cov : ndarray (n_components, n_components)
        Parameter covariance

    """
    # assert y.shape[-2] == yinvvar.shape[-2]
    assert y.shape[-1] == yinvvar.shape[-1]
    # assert y.shape[-1] == 1
    assert M_T.shape[-1] == yinvvar.shape[-1]
    assert np.all(np.isfinite(yinvvar))  # no negative elements
    assert np.all(np.isfinite(y))  # all finite
    assert np.all(np.isfinite(M_T))  # all finite
    assert np.count_nonzero(yinvvar) > 2  # at least two valid (non zero) pixels

    log2pi = np.log(2.0 * np.pi)
    nt = np.shape(M_T)[-2]
    ny = np.count_nonzero(yinvvar)
    M = np.transpose(M_T)  # (n_pix_y, n_components)
    Myinv = M * yinvvar[:, None]  # (n_pix_y, n_components)
    Hbar = np.matmul(M_T, Myinv)  #  (n_components, n_components)
    etabar = np.sum(Myinv * y[:, None], axis=0)  # (n_components)
    theta_map = np.linalg.solve(Hbar, etabar)  # (n_components)
    theta_cov = np.linalg.inv(Hbar)  # (n_components, n_components)
    if logyinvvar is None:
        logyinvvar = np.where(yinvvar == 0, 0, np.log(yinvvar))
    logdetH = np.sum(logyinvvar)  # scalar
    xi1 = -0.5 * (ny * log2pi - logdetH + np.sum(y * y * yinvvar))  # scalar
    sign, logdetHbar = np.linalg.slogdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


def logmarglike_lineargaussianmodel_onetransfer_jit(M_T, y, yinvvar, logyinvvar):
    """
    Fit linear model to one Gaussian data set, with no (=uniform) prior on the linear components.

    Parameters
    ----------
    y, yinvvar, logyinvvar : ndarray (n_pix_y)
        data and data inverse variances.
        Zeros will be ignored.
    M_T : ndarray (n_components, n_pix_y)
        design matrix of linear model

    Returns
    -------
    logfml : ndarray scalar
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (n_components)
        Best fit MAP parameters
    theta_cov : ndarray (n_components, n_components)
        Parameter covariance

    """
    log2pi = np.log(2.0 * np.pi)
    nt = np.shape(M_T)[-2]
    ny = np.count_nonzero(yinvvar)
    M = np.transpose(M_T)  # (n_pix_y, n_components)
    Myinv = M * yinvvar[:, None]  # (n_pix_y, n_components)
    Hbar = np.matmul(M_T, Myinv)  #  (n_components, n_components)
    etabar = np.sum(Myinv * y[:, None], axis=0)  # (n_components)
    theta_map = np.linalg.solve(Hbar, etabar)  # (n_components)
    theta_cov = np.linalg.inv(Hbar)  # (n_components, n_components)
    logdetH = np.sum(logyinvvar)  # scalar
    xi1 = -0.5 * (ny * log2pi - logdetH + np.sum(y * y * yinvvar))  # scalar
    sign, logdetHbar = np.linalg.slogdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


logmarglike_lineargaussianmodel_onetransfer_jitvmap = vmap(
    logmarglike_lineargaussianmodel_onetransfer_jit, in_axes=(0, 0, 0, 0)
)
logmarglike_lineargaussianmodel_onetransfer_jitvmap.__doc__ = """
    Fit linear model to a batch of Gaussian data sets,
    with no (=uniform) prior on the linear components.

    Parameters
    ----------
    y, yinvvar, logyinvvar : ndarray (nobj, n_pix_y)
        data and data inverse variances.
        Zeros will be ignored.
    M_T : ndarray (n_components, n_pix_y)
        design matrix of linear model

    Returns
    -------
    logfml : ndarray (nobj)
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, n_components)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, n_components, n_components)
        Parameter covariance

    """


def logmarglike_lineargaussianmodel_twotransfers(
    M_T,  #  (n_components, n_pix_y)
    y,  # (n_pix_y)
    yinvvar,  # (n_pix_y)
    mu,  # (n_components)
    muinvvar,  #  (n_components)
    logyinvvar=None,
    logmuinvvar=None,
):
    """
    Fit linear model to one Gaussian data sets, with Gaussian prior on linear components.

    Parameters
    ----------
    y, yinvvar : ndarray (n_pix_y)
        data and data inverse variances
    M_T : ndarray (n_components, n_pix_y)
        design matrix of linear model
    mu, muinvvar : ndarray (n_components)
        data and data variances for y

    Returns
    -------
    logfml : ndarray scalar
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (n_components)
        Best fit MAP parameters
    theta_cov : ndarray (n_components, n_components)
        Parameter covariance

    """
    log2pi = np.log(2.0 * np.pi)
    nt = np.shape(M_T)[-2]
    ny = np.count_nonzero(yinvvar)
    nm = np.count_nonzero(muinvvar)
    M = np.transpose(M_T)  # (n_pix_y, n_components)
    Myinv = M * yinvvar[:, None]  # (n_pix_y, n_components)
    Hbar = (
        np.matmul(M_T, Myinv) + np.eye(nt) * muinvvar[:, None]
    )  #  (n_components, n_components)
    etabar = np.sum(Myinv * y[:, None], axis=0) + mu * muinvvar  # (n_components)
    theta_map = np.linalg.solve(Hbar, etabar)  # (n_components)
    theta_cov = np.linalg.inv(Hbar)  # (n_components, n_components)
    if logyinvvar is None:
        logyinvvar = np.where(yinvvar == 0, 0, np.log(yinvvar))
    if logmuinvvar is None:
        logmuinvvar = np.where(muinvvar == 0, 0, np.log(muinvvar))
    logdetH = +np.sum(logyinvvar) + np.sum(logmuinvvar)  # scalar
    xi1 = -0.5 * (
        (ny + nm) * log2pi
        - logdetH
        + np.sum(y * y * yinvvar)
        + np.sum(mu * mu * muinvvar)
    )  # scalar
    sign, logdetHbar = np.linalg.slogdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    print("my Cinv_X", np.sum(y * y * yinvvar) - np.sum(etabar * theta_map))
    print("my logdet", -logdetH + logdetHbar)
    print("my counts", (ny + nm - nt) * log2pi)
    return logfml, theta_map, theta_cov


@jit
def logmarglike_lineargaussianmodel_twotransfers_jit(
    M_T,  #  (n_components, n_pix_y)
    y,  # (n_pix_y)
    yinvvar,  # (n_pix_y)
    logyinvvar,  # (n_pix_y)
    mu,  # (n_components)
    muinvvar,  #  (n_components)
    logmuinvvar,  # (n_components)
):
    """
    Fit linear model to one Gaussian data sets, with Gaussian prior on linear components.

    Parameters
    ----------
    y, yinvvar, logyinvvar : ndarray (n_pix_y)
        data and data inverse variances
    M_T : ndarray (n_components, n_pix_y)
        design matrix of linear model
    mu, muinvvar : ndarray (n_components)
        data and data variances for y

    Returns
    -------
    logfml : ndarray scalar
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (n_components)
        Best fit MAP parameters
    theta_cov : ndarray (n_components, n_components)
        Parameter covariance

    """
    log2pi = np.log(2.0 * np.pi)
    nt = np.shape(M_T)[-2]
    ny = np.count_nonzero(yinvvar)
    nm = np.count_nonzero(muinvvar)
    M = np.transpose(M_T)  # (n_pix_y, n_components)
    Myinv = M * yinvvar[:, None]  # (n_pix_y, n_components)
    Hbar = (
        np.matmul(M_T, Myinv) + np.eye(nt) * muinvvar[:, None]
    )  #  (n_components, n_components)
    etabar = np.sum(Myinv * y[:, None], axis=0) + mu * muinvvar  # (n_components)
    theta_map = np.linalg.solve(Hbar, etabar)  # (n_components)
    theta_cov = np.linalg.inv(Hbar)  # (n_components, n_components)
    logdetH = +np.sum(logyinvvar) + np.sum(logmuinvvar)  # scalar
    xi1 = -0.5 * (
        (ny + nm) * log2pi
        - logdetH
        + np.sum(y * y * yinvvar)
        + np.sum(mu * mu * muinvvar)
    )  # scalar
    sign, logdetHbar = np.linalg.slogdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


logmarglike_lineargaussianmodel_twotransfers_jitvmap = vmap(
    logmarglike_lineargaussianmodel_twotransfers_jit,
    in_axes=(0, 0, 0, 0, 0, 0, 0),
)


def logmarglike_lineargaussianmodel_threetransfers(
    ell,  # scalar
    M_T,  #  (n_components, n_pix_y)
    R_T,  # (n_components, n_pix_z)
    y,  # (n_pix_y)
    yinvvar,  # (n_pix_y)
    z,  #  (n_pix_z)
    zinvvar,  #  (n_pix_z)
    mu,  # (n_components)
    muinvvar,  #  (n_components)
    logyinvvar=None,
    logzinvvar=None,
    logmuinvvar=None,
):
    """
    Fit linear model to two Gaussian data sets, with Gaussian prior on components.

    Parameters
    ----------
    ell : ndarray scalar
        scaling between the data: y = ell * z
    y, yinvvar : ndarray (n_pix_y)
        data and data inverse variances
    M_T : ndarray (n_components, n_pix_y)
        design matrix of linear model
    z, zinvvar : ndarray (n_pix_z)
        data and data variances for y
    R_T : ndarray (n_components, n_pix_z)
        design matrix of linear model for z
    mu, muinvvar : ndarray (n_components)
        data and data variances for y

    Returns
    -------
    logfml : ndarray scalar
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (n_components)
        Best fit MAP parameters
    theta_cov : ndarray (n_components, n_components)
        Parameter covariance

    """
    log2pi = np.log(2.0 * np.pi)
    nt = np.shape(M_T)[-2]
    ny = np.count_nonzero(yinvvar)
    nz = np.count_nonzero(zinvvar)
    nm = np.count_nonzero(muinvvar)
    M = np.transpose(M_T)  # (n_pix_y, n_components)
    R = np.transpose(R_T)  # (n_pix_z, n_components)
    Myinv = M * yinvvar[:, None]  # (n_pix_y, n_components)
    Rzinv = R * zinvvar[:, None]  # (n_pix_z, n_components)
    Hbar = (
        ell ** 2 * np.matmul(R_T, Rzinv)
        + np.matmul(M_T, Myinv)
        + np.eye(nt) * muinvvar[:, None]
    )  #  (n_components, n_components)
    etabar = (
        ell * np.sum(Rzinv * z[:, None], axis=0)
        + np.sum(Myinv * y[:, None], axis=0)
        + mu * muinvvar
    )  # (n_components)
    theta_map = np.linalg.solve(Hbar, etabar)  # (n_components)
    theta_cov = np.linalg.inv(Hbar)  # (n_components, n_components)
    if logyinvvar is None:
        logyinvvar = np.where(yinvvar == 0, 0, np.log(yinvvar))
    if logzinvvar is None:
        logzinvvar = np.where(zinvvar == 0, 0, np.log(zinvvar))
    if logmuinvvar is None:
        logmuinvvar = np.where(muinvvar == 0, 0, np.log(muinvvar))
    logdetH = np.sum(logyinvvar) + np.sum(logzinvvar) + np.sum(logmuinvvar)  # scalar
    xi1 = -0.5 * (
        (ny + nz + nm) * log2pi
        - logdetH
        + np.sum(y * y * yinvvar)
        + np.sum(z * z * zinvvar)
        + np.sum(mu * mu * muinvvar)
    )  # scalar
    sign, logdetHbar = np.linalg.slogdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


@jit
def logmarglike_lineargaussianmodel_threetransfers_jit(
    ell,  # scalar
    M_T,  #  (n_components, n_pix_y)
    R_T,  # (n_components, n_pix_z)
    y,  # (n_pix_y)
    yinvvar,  # (n_pix_y),
    logyinvvar,  # (n_pix_y),
    z,  #  (n_pix_z)
    zinvvar,  #  (n_pix_z)
    logzinvvar,  #  (n_pix_z)
    mu,  # (n_components)
    muinvvar,  # (n_components)
    logmuinvvar,  # (n_components)
):
    """
    Fit linear model to two Gaussian data sets, with Gaussian prior on components.

    Parameters
    ----------
    ell : ndarray scalar
        scaling between the data: y = ell * z
    y, yinvvar, logyinvvar : ndarray (n_pix_y)
        data and data inverse variances
    M_T : ndarray (n_components, n_pix_y)
        design matrix of linear model
    z, zinvvar, logzinvvar : ndarray (n_pix_z)
        data and data variances for y
    R_T : ndarray (n_components, n_pix_z)
        design matrix of linear model for z
    mu, muinvvar, logmuinvvar : ndarray (n_components)
        data and data variances for y

    Returns
    -------
    logfml : ndarray scalar
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (n_components)
        Best fit MAP parameters
    theta_cov : ndarray (n_components, n_components)
        Parameter covariance

    """
    log2pi = np.log(2.0 * np.pi)
    nt = np.shape(M_T)[-2]
    ny = np.count_nonzero(yinvvar)
    nz = np.count_nonzero(zinvvar)
    nm = np.count_nonzero(muinvvar)
    M = np.transpose(M_T)  # (n_pix_y, n_components)
    R = np.transpose(R_T)  # (n_pix_z, n_components)
    Myinv = M * yinvvar[:, None]  # (n_pix_y, n_components)
    Rzinv = R * zinvvar[:, None]  # (n_pix_z, n_components)
    Hbar = (
        ell ** 2 * np.matmul(R_T, Rzinv)
        + np.matmul(M_T, Myinv)
        + np.eye(nt) * muinvvar[:, None]
    )  #  (n_components, n_components)
    etabar = (
        ell * np.sum(Rzinv * z[:, None], axis=0)
        + np.sum(Myinv * y[:, None], axis=0)
        + mu * muinvvar
    )  # (n_components)
    theta_map = np.linalg.solve(Hbar, etabar)  # (n_components)
    theta_cov = np.linalg.inv(Hbar)  # (n_components, n_components)
    logdetH = np.sum(logyinvvar) + np.sum(logzinvvar) + np.sum(logmuinvvar)  # scalar
    xi1 = -0.5 * (
        (ny + nz + nm) * log2pi
        - logdetH
        + np.sum(y * y * yinvvar)
        + np.sum(z * z * zinvvar)
        + np.sum(mu * mu * muinvvar)
    )  # scalar
    sign, logdetHbar = np.linalg.slogdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


logmarglike_lineargaussianmodel_threetransfers_jitvmap = vmap(
    logmarglike_lineargaussianmodel_threetransfers_jit,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)


@jit
def logmarglike_scalingmodel_flatprior_jit(
    ymod,  #  (n_components, n_pix_y)
    y,  # (n_components, n_pix_y)
    yinvvar,  # (n_components, n_pix_y)
    logyinvvar,  # (n_components, n_pix_y)
):
    """
    Fit model to one Gaussian data set, with Gaussian prior on scaling.

    Parameters
    ----------
    y, yinvvar, logyinvvar : ndarray (n_components, n_pix_y)
        data and data inverse variances
    ymod : ndarray (n_components, n_pix_y)
        design matrix of linear model
    mu, muinvvar : ndarray (n_components)
        priors

    Returns
    -------
    logfml : ndarray (n_components)
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (n_components)
        Best fit MAP parameters
    theta_cov : ndarray (n_components)
        Parameter covariance

    """
    log2pi = np.log(2.0 * np.pi)
    ny = np.count_nonzero(yinvvar)
    # n_components
    FOT = np.sum(ymod * y * yinvvar, axis=-1)
    FTT = np.sum(ymod ** 2 * yinvvar, axis=-1)
    FOO = np.sum(ymod ** 2 * yinvvar, axis=-1)
    logSigma_det = np.sum(logyinvvar, axis=-1)
    ellML = FOT / FTT
    chi2 = FOO - (FOT / FTT) * FOT
    logfml = -0.5 * (chi2 + np.log(FTT) - logSigma_det + ny * log2pi)
    theta_map = FOT / FTT
    theta_cov = FTT ** -1
    return logfml, theta_map, theta_cov


@jit
def logmarglike_scalingmodel_gaussianprior(
    ymod,  #  (n_components, n_pix_y)
    y,  # (n_components, n_pix_y)
    yinvvar,  # (n_components, n_pix_y)
    logyinvvar,  # (n_components, n_pix_y)
    mu,  # (n_components)
    muinvvar,  #  (n_components)
    logmuinvvar,  # (n_components)
):
    (
        logfml,
        theta_map,
        theta_cov,
    ) = logmarglike_lineargaussianmodel_twotransfers_jitvmap(
        ymod[:, None, :],
        y[:, :],
        yinvvar[:, :],
        logyinvvar[:, :],
        mu[:, None],
        muinvvar[:, None],
        logmuinvvar[:, None],
    )
    return np.squeeze(logfml), np.squeeze(theta_map), np.squeeze(theta_cov)


logmarglike_scalingmodel_gaussianprior_jitvmap = jit(
    vmap(
        logmarglike_scalingmodel_gaussianprior,
        in_axes=(0, 0, 0, 0, 0, 0, 0),
    )
)

logmarglike_lineargaussianmodel_onetransfer_jitvmapvmap = jit(
    vmap(logmarglike_lineargaussianmodel_onetransfer_jitvmap)
)
logmarglike_lineargaussianmodel_onetransfer_jitvmapvmap.__doc__ = """
    Fit linear model to a batch of Gaussian data sets,
    with no (=uniform) prior on the linear components.

    Parameters
    ----------
    ymod,  #  (nobj, nt, n_components, n_pix_y)
    y,  # (nobj, nt, n_pix_y)
    yinvvar,  # (nobj, nt, n_pix_y)
    logyinvvar,  # (nobj, nt, n_pix_y)

    Returns
    -------
    logfml : ndarray (nobj, nt)
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, nt, n_components)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, nt, n_components, n_components)
        Parameter covariance

    """


logmarglike_lineargaussianmodel_twotransfers_jitvmapvmap = jit(
    vmap(logmarglike_lineargaussianmodel_twotransfers_jitvmap)
)
logmarglike_lineargaussianmodel_twotransfers_jitvmapvmap.__doc__ = """

    Parameters
    ----------
        ymod,  #  (nobj, nt, n_components, n_pix_y)
        y,  # (nobj, nt, n_pix_y)
        yinvvar,  # (nobj, nt, n_pix_y)
        logyinvvar,  # (nobj, nt, n_pix_y)
        mu,  # (nobj, nt, n_components)
        muinvvar,  #  (nobj, nt, n_components)
        logmuinvvar,  # (nobj, nt, n_components)

    Returns
    -------
    logfml : ndarray (nobj, nt)
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, nt, n_components)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, nt, n_components, n_components)
        Parameter covariance

    """


logmarglike_lineargaussianmodel_threetransfers_jitvmapvmap = jit(
    vmap(logmarglike_lineargaussianmodel_threetransfers_jitvmap)
)
logmarglike_lineargaussianmodel_threetransfers_jitvmapvmap.__doc__ = """

    Parameters
    ----------
    ell,  # (nobj, nt, )
    ymod,  #  (nobj, nt, n_components, n_pix_y)
    zmod,  # (nobj, nt, n_components, n_pix_z)
    y,  # (nobj, nt, n_pix_y)
    yinvvar,  # (nobj, nt, n_pix_y),
    logyinvvar,  # (nobj, nt, n_pix_y),
    z,  #  (nobj, nt, n_pix_z)
    zinvvar,  #  (nobj, nt, n_pix_z)
    logzinvvar,  #  (nobj, nt, n_pix_z)
    mu,  # (nobj, nt, n_components)
    muinvvar,  # (nobj, nt, n_components)
    logmuinvvar,  # (nobj, nt, n_components)

    Returns
    -------
    logfml : ndarray (nobj, nt)
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, nt, n_components)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, nt, n_components, n_components)
        Parameter covariance

    """
