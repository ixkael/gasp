# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import jit, vmap


def logmarglike_lineargaussianmodel_onetransfer(M_T, y, yinvvar, logyinvvar=None):
    """
    Fit linear model to one Gaussian data set, with no (=uniform) prior on the linear components.

    Parameters
    ----------
    y, yinvvar : ndarray (n_pix_y)
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
    logyinvvar : TODO

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
    xi2 = -0.5 * (nt * log2pi - sign * logdetHbar + np.sum(etabar * theta_map))
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
    xi2 = -0.5 * (nt * log2pi - sign * logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


logmarglike_lineargaussianmodel_onetransfer_jitvmap = vmap(
    logmarglike_lineargaussianmodel_onetransfer_jit, in_axes=(None, 0, 0, 0)
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
    xi2 = -0.5 * (nt * log2pi - sign * logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


@jit
def logmarglike_lineargaussianmodel_twotransfers_jit(
    M_T,  #  (n_components, n_pix_y)
    y,  # (n_pix_y)
    yinvvar,  # (n_pix_y)
    logyinvvar,  # (n_pix_y)
    mu,  # (n_components)
    muinvvar,  #  (n_components)
    logmuinvvar,  # (n_pix_y)
):
    """
    Fit linear model to one Gaussian data sets, with Gaussian prior on linear components.

    Parameters
    ----------
    y, yinvvar, logyinvvar : ndarray (n_pix_y)
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
    xi2 = -0.5 * (nt * log2pi - sign * logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


logmarglike_lineargaussianmodel_twotransfers_jitvmap = vmap(
    logmarglike_lineargaussianmodel_twotransfers_jit,
    in_axes=(None, 0, 0, 0, 0, 0, 0),
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
    xi2 = -0.5 * (nt * log2pi - sign * logdetHbar + np.sum(etabar * theta_map))
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
    xi2 = -0.5 * (nt * log2pi - sign * logdetHbar + np.sum(etabar * theta_map))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


logmarglike_lineargaussianmodel_threetransfers_jitvmap = vmap(
    logmarglike_lineargaussianmodel_threetransfers_jit,
    in_axes=(0, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)
