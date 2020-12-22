# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np

T = tf.float64

log2pi = tf.cast(tf.math.log(2.0 * np.pi), T)


def int_log2pi(k):
    return tf.cast(k, T) * tf.cast(tf.math.log(2.0 * np.pi), T)


def loggaussian(x, mean=tf.cast(0.0, T), std=tf.cast(1.0, T)):
    return -0.5 * tf.reduce_sum(
        tf.square((x - mean) / std) + log2pi + 2 * tf.math.log(std), axis=-1
    )


def lnlike_ellflatpriormarginalized_multiple_withprior(
    y,  # (..., dy)
    yvar,  # (..., dy)
    mods_T,  #  (..., dt, dy),
    mods_mean,  #  (..., dt), mu
    mods_var,  #  (..., dt), lambda
    expand_dims,  #  (..., dt, dt), lambda
    compute_marglike=True,
):

    nmod = tf.shape(mods_mean)[-1]
    ndata = tf.shape(y)[-1]
    mods = tf.transpose(mods_T, [0, 1, 3, 2])  # tf.einsum("...ij->...ji", mods_T)

    t = expand_dims * tf.eye(nmod, dtype=T)[None, None, :, :]
    mods_covar_inv = t * tf.expand_dims(1 / mods_var, -1)

    A = mods_covar_inv + tf.matmul(mods_T, mods / yvar[..., :, None])  #  (..., dt, dt)

    ymodprior = tf.reduce_sum(mods_T * tf.expand_dims(mods_mean, -1), -2)
    d = y - ymodprior
    e = tf.reduce_sum(mods_T * (d / yvar)[..., None, :], axis=-1)

    Ainve = tf.linalg.solve(A, e[..., None])[..., 0]  # (..., dt)

    chi2 = tf.reduce_sum(d * d / yvar, axis=-1) - tf.reduce_sum(e * Ainve, axis=-1)

    s, logdetA = tf.linalg.slogdet(A)
    ldets = (
        logdetA
        + tf.reduce_sum(tf.math.log(mods_var), axis=-1)
        + tf.reduce_sum(tf.math.log(yvar), axis=-1)
    )
    scalar = tf.cast(ndata, T) * log2pi
    LnMarglike = -0.5 * (scalar + ldets + chi2)

    eta = mods_mean / mods_var + tf.reduce_sum(
        mods_T * (y / yvar)[..., None, :], axis=-1
    )  # (..., dt)
    mu = tf.linalg.solve(A, eta[..., None])[..., 0]  # (..., dt)

    return LnMarglike, mu


def lnlike_ellflatpriormarginalized(
    F_obs,  # (nobj, ..., numBands)
    F_obs_var,  # (nobj, ..., numBands)
    F_mod,  #  (nobj, ..., numBands)
):
    """
    Fit linear model to one Gaussian data set (formulation 3)

    Parameters
    ----------
    F_obs, F_obs_var : ndarray (nobj, ..., n_pix_y)
        data and data variances
    F_mod : ndarray (..., n_components, n_pix_y)
        design matrix of linear model

    Returns
    -------
    logfml : ndarray (nobj, )
        log likelihood values with parameters marginalised and at best fit
    ellML : ndarray (nobj, ndim)
        Best fit MAP parameters

    """
    FOT = tf.reduce_sum(F_mod * F_obs / F_obs_var, axis=-1)  # (nobj, ..., )
    FOO = tf.reduce_sum(tf.square(F_obs) / F_obs_var, axis=-1)  # (nobj, ..., )
    FTT = tf.reduce_sum(tf.square(F_mod) / F_obs_var, axis=-1)  # (nobj, ..., )
    LogSigma_det = tf.reduce_sum(tf.math.log(F_obs_var), axis=-1)  # (nobj, ..., )
    Chi2 = FOO - tf.multiply(tf.divide(FOT, FTT), FOT)  # (nobj, ..., )
    LogDenom = LogSigma_det + tf.math.log(FTT)
    LnMarglike = -0.5 * Chi2 - 0.5 * LogDenom  # (nobj, ..., )
    ellML = FOT / FTT
    return LnMarglike, ellML


def lnlike_ellflatpriormarginalized_multiple(
    y, yinvvar, mods  # (..., dy)  # (..., dy)  #  (..., dt, dy)
):
    """
    Fit linear model to one Gaussian data set (formulation 1)

    Parameters
    ----------
    y, yinvvar : ndarray (nobj, ..., n_pix_y)
        data and data inverse variances
    M_T : ndarray (..., n_components, n_pix_y)
        design matrix of linear model

    Returns
    -------
    logfml : ndarray (nobj, )
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, ndim)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, ndim, ndim)
        Parameter covariance

    """
    eta = tf.reduce_sum(mods * (y * yinvvar)[..., None, :], axis=-1)  # (..., dt)
    H = tf.matmul(
        mods, tf.transpose(mods * yinvvar[..., None, :], [0, 1, 3, 2])
    )  # (..., dt, dt)
    mu = tf.linalg.solve(H, eta[..., None])[..., 0]  # (..., dt)
    etaHinveta = tf.reduce_sum(eta * mu, axis=-1)  # (..., )
    yyvarinvy = tf.reduce_sum(y * y * yinvvar, axis=-1)  # (..., )
    dets = tf.linalg.logdet(H) - tf.reduce_sum(
        tf.where(yinvvar > 0, tf.math.log(yinvvar), yinvvar * 0), axis=-1
    )
    scalar = tf.cast(tf.shape(mods)[-1] - tf.shape(mods)[-2], T) * log2pi
    LnMarglike = -0.5 * (scalar + dets + yyvarinvy - etaHinveta)
    covar = tf.linalg.inv(H)
    return LnMarglike, mu, covar


def logmarglike_onetransfergaussian(y, yinvvar, M_T):  # (..., dy)  # (..., dy)
    """
    Fit linear model to one Gaussian data set (formulation 2)

    Parameters
    ----------
    y, yinvvar : ndarray (nobj, ..., n_pix_y)
        data and data inverse variances
    M_T : ndarray (..., n_components, n_pix_y)
        design matrix of linear model

    Returns
    -------
    logfml : ndarray (nobj, )
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, ndim)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, ndim, ndim)
        Parameter covariance

    """

    nt = tf.cast(tf.shape(M_T)[-2], T)
    ny = tf.cast(tf.math.count_nonzero(tf.where(yinvvar > 0)), T)
    M = tf.transpose(M_T, [0, 2, 1])  # tf.einsum("...ij->...ji", M_T)
    Hbar = tf.matmul(M_T, M * yinvvar[..., :, None])  #  (..., dt, dt)
    etabar = tf.reduce_sum(M_T * (y * yinvvar)[..., None, :], axis=-1)  # (..., dt)
    theta_map = tf.linalg.solve(Hbar, etabar[..., None])[..., 0]  # (..., dt)
    theta_cov = tf.linalg.inv(Hbar)
    xi1 = -0.5 * (
        ny * log2pi
        + tf.reduce_sum(y * y * yinvvar, axis=-1)
        - tf.reduce_sum(
            tf.where(yinvvar > 0, tf.math.log(yinvvar), yinvvar * 0), axis=-1
        )
    )
    logdetHbar = tf.linalg.logdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + tf.reduce_sum(etabar * theta_map, axis=-1))
    logfml = xi1 - xi2

    return logfml, theta_map, theta_cov


def logmarglike_twotransfergaussians(
    ells,
    y,  # (..., dy)
    yinvvar,  # (..., dy)
    M_T,  #  (..., dt, dy),
    z,  #  (..., dz)
    zinvvar,  #  (..., dz)
    R_T,  #  (..., dt, dz),
    perm=[0, 2, 1],
):
    """
    Fit linear model to two Gaussian data sets

    Parameters
    ----------
    ells : ndarray (nobj, )
        scaling between the data: y = ell * z
    y, yinvvar : ndarray (nobj, ..., n_pix_y)
        data and data inverse variances
    M_T : ndarray (..., n_components, n_pix_y)
        design matrix of linear model
    z, zinvvar : ndarray (nobj, ..., n_pix_z)
        data and data inverse variances for z
    R_T : ndarray (..., n_components, n_pix_z)
        design matrix of linear model for z
    perm : list
        permutation to get M and R from R_T and M_T

    Returns
    -------
    logfml : ndarray (nobj, )
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, ndim)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, ndim, ndim)
        Parameter covariance

    """
    log2pi = tf.cast(tf.math.log(2.0 * np.pi), T)
    nt = tf.cast(tf.shape(M_T)[-2], T)
    ny = tf.cast(
        tf.math.count_nonzero(tf.where(yinvvar > 0)), T
    )  # tf.cast(tf.shape(y)[-1], T)
    nz = tf.cast(
        tf.math.count_nonzero(tf.where(zinvvar > 0)), T
    )  # tf.cast(tf.shape(z)[-1], T)
    M = tf.transpose(M_T, perm)  # tf.einsum("...ij->...ji", M_T)
    R = tf.transpose(R_T, perm)  # tf.einsum("...ij->...ji", M_T)
    Hbar = ells[..., None, None] ** 2 * tf.matmul(
        R_T, R * zinvvar[..., :, None]
    ) + tf.matmul(
        M_T, M * yinvvar[..., :, None]
    )  #  (..., dt, dt)
    etabar = ells[..., None] * tf.reduce_sum(
        R_T * (z * zinvvar)[..., None, :], axis=-1
    ) + tf.reduce_sum(
        M_T * (y * yinvvar)[..., None, :], axis=-1
    )  # (..., dt)
    theta_map = tf.linalg.solve(Hbar, etabar[..., None])[..., 0]  # (..., dt)
    theta_cov = tf.linalg.inv(Hbar)
    logdetH = tf.reduce_sum(
        tf.where(zinvvar > 0, tf.math.log(zinvvar), zinvvar * 0), axis=-1
    ) + tf.reduce_sum(tf.where(yinvvar > 0, tf.math.log(yinvvar), yinvvar * 0), axis=-1)
    xi1 = -0.5 * (
        (ny + nz) * log2pi
        - logdetH
        + tf.reduce_sum(y * y * yinvvar, axis=-1)
        + tf.reduce_sum(z * z * zinvvar, axis=-1)
    )
    logdetHbar = tf.linalg.logdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + tf.reduce_sum(etabar * theta_map, axis=-1))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


def logmarglike_threetransfergaussians(
    ells,  # (..., )
    y,  # (..., dy)
    yinvvar,  # (..., dy)
    M_T,  #  (..., dt, dy),
    z,  #  (..., dz),
    zinvvar,  #  (..., dz),
    R_T,  #  (..., dt, dz),
    mu,  #  (..., dt),
    muinvvar,  #  (..., dt),
):
    """
    Fit linear model to three Gaussian data sets

    Parameters
    ----------
    ells : ndarray (nobj, )
        scaling between the data: y = ell * z
    y, yinvvar : ndarray (nobj, ..., n_pix_y)
        data and data inverse variances
    M_T : ndarray (..., n_components, n_pix_y)
        design matrix of linear model
    z, zinvvar : ndarray (nobj, ..., n_pix_z)
        data and data variances for y
    R_T : ndarray (..., n_components, n_pix_z)
        design matrix of linear model for z
    mu, muinvvar : ndarray ( ..., n_components)
        data and data variances for y

    Returns
    -------
    logfml : ndarray (nobj, )
        log likelihood values with parameters marginalised and at best fit
    theta_map : ndarray (nobj, ndim)
        Best fit MAP parameters
    theta_cov : ndarray (nobj, ndim, ndim)
        Parameter covariance

    """
    log2pi = tf.cast(tf.math.log(2.0 * np.pi), T)
    nt = tf.cast(tf.shape(M_T)[-2], T)
    nobj = tf.cast(tf.shape(y)[0], T)
    ny = tf.cast(
        tf.math.count_nonzero(tf.where(yinvvar > 0)), T
    )  # tf.cast(tf.shape(y)[-1], T)
    nz = tf.cast(
        tf.math.count_nonzero(tf.where(zinvvar > 0)), T
    )  # tf.cast(tf.shape(z)[-1], T)
    nm = tf.cast(
        tf.math.count_nonzero(tf.where(muinvvar > 0)), T
    )  # tf.cast(tf.shape(mu)[-1], T)
    M = tf.transpose(M_T, [0, 2, 1])  # tf.einsum("...ij->...ji", M_T)
    R = tf.transpose(R_T, [0, 2, 1])  # tf.einsum("...ij->...ji", M_T)
    Hbar = (
        ells[:, None, None] ** 2 * tf.matmul(R_T, R * zinvvar[..., :, None])
        + tf.matmul(M_T, M * yinvvar[..., :, None])
        + tf.eye(nt, dtype=T)[None, :, :]
        * tf.ones((nobj, 1, 1), dtype=T)
        * muinvvar[..., :, None]
    )  #  (..., dt, dt)
    etabar = (
        ells[:, None] * tf.reduce_sum(R_T * (z * zinvvar)[..., None, :], axis=-1)
        + tf.reduce_sum(M_T * (y * yinvvar)[..., None, :], axis=-1)
        + tf.reduce_sum((mu * muinvvar)[..., None, :], axis=-1)
    )  # (..., dt)
    theta_map = tf.linalg.solve(Hbar, etabar[..., None])[..., 0]  # (..., dt)
    theta_cov = tf.linalg.inv(Hbar)
    logdetH = (
        tf.reduce_sum(tf.where(zinvvar > 0, tf.math.log(zinvvar), zinvvar * 0), axis=-1)
        + tf.reduce_sum(
            tf.where(yinvvar > 0, tf.math.log(yinvvar), yinvvar * 0), axis=-1
        )
        + tf.reduce_sum(
            tf.where(muinvvar > 0, tf.math.log(muinvvar), muinvvar * 0), axis=-1
        )
    )
    xi1 = -0.5 * (
        (ny + nz + nm) * log2pi
        - logdetH
        + tf.reduce_sum(y * y * yinvvar, axis=-1)
        + tf.reduce_sum(z * z * zinvvar, axis=-1)
        + tf.reduce_sum(mu * mu * muinvvar, axis=-1)
    )
    logdetHbar = tf.linalg.logdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + tf.reduce_sum(etabar * theta_map, axis=-1))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


def logmarglike_threetransfergaussians_secondfull(
    ells,  # (..., )
    y,  # (..., dy)
    yvar,  # (..., dy)
    M_T,  #  (..., dt, dy),
    z,  #  (..., dz),
    zcovar,  #  (..., dz, dz),
    R_T,  #  (..., dt, dz),
    mu,  #  (..., dt),
    muvar,  #  (..., dt),
):

    nt = tf.cast(tf.shape(M_T)[-2], T)
    nobj = tf.cast(tf.shape(y)[0], T)
    ny = tf.cast(tf.shape(y)[-1], T)
    nz = tf.cast(tf.shape(z)[-1], T)
    nm = tf.cast(tf.shape(mu)[-1], T)
    M = tf.transpose(M_T, [0, 2, 1])  # tf.einsum("...ij->...ji", M_T)
    R = tf.transpose(R_T, [0, 2, 1])  # tf.einsum("...ij->...ji", M_T)
    zvarinvz = tf.linalg.solve(zcovar, z[:, :, None])
    Hbar = (
        ells[:, None, None] ** 2 * tf.matmul(R_T, tf.linalg.solve(zcovar, R))
        + tf.matmul(M_T, M / yvar[..., :, None])
        + tf.eye(nt, dtype=T)[None, :, :]
        * tf.ones((nobj, 1, 1), dtype=T)
        / muvar[..., :, None]
    )  #  (..., dt, dt)
    etabar = (
        ells[:, None]
        * tf.reduce_sum(R_T[..., None] * zvarinvz[..., None, :, :], axis=(-2, -1))
        + tf.reduce_sum(M_T * (y / yvar)[..., None, :], axis=-1)
        + tf.reduce_sum((mu / muvar)[..., None, :], axis=-1)
    )  # (..., dt)
    theta_map = tf.linalg.solve(Hbar, etabar[..., None])[..., 0]  # (..., dt)
    theta_cov = tf.linalg.inv(Hbar)
    logdetH = (
        -tf.linalg.logdet(zcovar)
        - tf.reduce_sum(tf.math.log(yvar), axis=-1)
        - tf.reduce_sum(tf.math.log(muvar), axis=-1)
    )
    xi1 = -0.5 * (
        (ny + nz + nm) * log2pi
        - logdetH
        + tf.reduce_sum(y * y / yvar, axis=-1)
        + tf.reduce_sum(z[:, None] * zvarinvz, axis=(-1, -2))
        + tf.reduce_sum(mu * mu / muvar, axis=-1)
    )
    logdetHbar = tf.linalg.logdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + tf.reduce_sum(etabar * theta_map, axis=-1))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov


def solve_woodbury_sum_onefull_oneinv(A_full, Binv_diag, target):
    n_B = tf.shape(Binv_diag)[1]
    Binv_target = Binv_diag * target
    # (A+B)inv * target = (Binv * target) - Binv * (Ainv + Binv)inv * (Binv * target)
    Ainv = tf.linalg.inv(A_full)
    Binv_full = tf.eye(n_B, dtype=T)[None, ...] * Binv_diag
    Ainv_plus_Binv = Ainv + Binv_full
    temp = tf.linalg.solve(Ainv_plus_Binv, Binv_target)
    print(tf.shape(Ainv_plus_Binv), tf.shape(temp))
    corr = tf.matmul(Binv_full, temp)
    return Binv_target - corr


def logmarglike_twotransfergaussians_fullrank(
    y,  # (..., dy)
    ycovar,  # (..., dy)
    yinvvar,
    M_T,  #  (..., dt, dy),
    z,  #  (..., dz), mu
    zinvvar,  #  (..., dz), lambda,
    R_T,  #  (..., dt, dz),
    perm=[0, 2, 1],
):
    log2pi = tf.cast(tf.math.log(2.0 * np.pi), T)
    nt = tf.cast(tf.shape(M_T)[-2], T)
    ny = tf.cast(tf.shape(y)[-1], T)
    nz = tf.cast(
        tf.math.count_nonzero(tf.where(zinvvar > 0)), T
    )  # tf.cast(tf.shape(z)[-1], T)
    M = tf.transpose(M_T, perm)  # tf.einsum("...ij->...ji", M_T)
    R = tf.transpose(R_T, perm)  # tf.einsum("...ij->...ji", M_T)
    # XinvM = tf.linalg.solve(ycovar, M)
    # Xinvy = tf.linalg.solve(ycovar, y[..., None])
    XinvM = solve_woodbury_sum_onefull_oneinv(ycovar, yinvvar[..., None], M)
    Xinvy = solve_woodbury_sum_onefull_oneinv(ycovar, yinvvar[..., None], y[..., None])
    Hbar = tf.matmul(R_T, R * zinvvar[..., :, None]) + tf.matmul(
        M_T, XinvM
    )  #  (..., dt, dt)
    etabar = tf.reduce_sum(R_T * (z * zinvvar)[..., None, :], axis=-1) + tf.reduce_sum(
        M_T[..., None] * Xinvy[:, None, :, :], axis=(-2, -1)
    )  # (..., dt)
    theta_map = tf.linalg.solve(Hbar, etabar[..., None])[..., 0]  # (..., dt)
    theta_cov = tf.linalg.inv(Hbar)
    logdetH = tf.reduce_sum(
        tf.where(zinvvar > 0, tf.math.log(zinvvar), zinvvar * 0), axis=-1
    ) - tf.linalg.logdet(ycovar)
    xi1 = -0.5 * (
        (ny + nz) * log2pi
        - logdetH
        + tf.reduce_sum(y[:, None] * Xinvy, axis=(-1, -2))
        + tf.reduce_sum(z * z * zinvvar, axis=-1)
    )
    logdetHbar = tf.linalg.logdet(Hbar)
    xi2 = -0.5 * (nt * log2pi - logdetHbar + tf.reduce_sum(etabar * theta_map, axis=-1))
    logfml = xi1 - xi2
    return logfml, theta_map, theta_cov, XinvM, Xinvy
