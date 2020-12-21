# -*- coding: utf-8 -*-

import numpy as np


def draw_uniform(samples, desired_size, bins=40):
    """
    Draw a set of elements from an array, such that the distribution is approximately uniform.

    Parameters
    ----------
    samples: ndarray (nobj, )
        Array of properties
    desired_size: int
        Number of objects to draw
    bins: int
        Number of bins to use for the drawn (default: 40)

    Returns
    -------
    indices: ndarray (approx_desired_size, )
        Set of indices drawn from initial array, approximately of size desired_size

    """
    hist, bin_edges = np.histogram(samples, bins=bins)
    avg_nb = int(desired_size / float(bins))
    numbers = np.repeat(avg_nb, bins)
    for j in range(4):
        numbers[hist <= numbers] = hist[hist <= numbers]
        nb_rest = desired_size - np.sum(numbers[hist <= numbers])  # * bins
        avg_nb = round(nb_rest / np.sum(hist > numbers))
        numbers[hist > numbers] = avg_nb

    result = []
    count = 0
    for i in range(bin_edges.size - 1):
        ind = samples >= bin_edges[i]
        ind &= samples <= bin_edges[i + 1]
        if ind.sum() > 0:
            positions = np.where(ind)[0]
            nb = np.min([numbers[i], ind.sum()])
            result.append(np.random.choice(positions, nb, replace=False))

    indices = np.concatenate(result)

    return indices


def generate_sample_grid(theta_mean, theta_std, n):
    """
    Create a meshgrid of n ** n_dim samples,
    tiling [theta_mean[i] - 5 * theta_std[i], theta_mean[i] + 5 * theta_std]
    into n portions.
    Also returns the volume element.

    Parameters
    ----------
    theta_mean, theta_std : ndarray (n_dim)

    Returns
    -------
    theta_samples : ndarray (nobj, n_dim)

    vol_element: scalar
        Volume element

    """
    n_components = theta_mean.size
    xs = [
        np.linspace(
            theta_mean[i] - 5 * theta_std[i],
            theta_mean[i] + 5 * theta_std[i],
            n,
        )
        for i in range(n_components)
    ]
    mxs = np.meshgrid(*xs)
    orshape = mxs[0].shape
    mxsf = np.vstack([i.ravel() for i in mxs]).T
    dxs = np.vstack([np.diff(xs[i])[i] for i in range(n_components)])
    vol_element = np.prod(dxs)
    theta_samples = np.vstack(mxsf)
    return theta_samples, vol_element


def make_masked_noisy_data(y_truth, masked_fraction=0.5):
    """
    From some noiseless data, generates some arbitrary noise and mask.

    Parameters
    ----------
    y_truth : ndarray (nobj, npix)
        Input noiseless data

    masked_fraction: scalar
        Fraction of data to mask (default: 0.5)

    Returns
    -------
    y_withzeros, yinvvar_withzeros, logyinvvar_withzeros : ndarray (..., npix)
        output noisy data, inverse variance, and log inverse variance,
        with masked pixels (= zeros in yinvvar_withzeros and logyinvvar_withzeros)

    """
    nobj, n_pix_y = y_truth.shape
    rn = jax.random.normal(key, (nobj, n_pix_y))
    yinvvar = 10 ** rn
    rn = jax.random.normal(key, (nobj, n_pix_y))
    y = y_truth + rn * (yinvvar ** -0.5)
    rn = onp.random.uniform(size=n_pix_y * nobj).reshape((nobj, n_pix_y))
    mask_y = rn < masked_fraction
    yinvvar_withzeros = yinvvar * mask_y
    y_withzeros = y * mask_y
    logyinvvar_withzeros = np.where(
        yinvvar_withzeros == 0, 0, np.log(yinvvar_withzeros)
    )
    return y_withzeros, yinvvar_withzeros, logyinvvar_withzeros


def batch_gaussian_loglikelihood(dx, x_invvar):
    """
    log of gaussian likelihood function for a batch of data.
    IMPORTANTLY, zeros in x_invvar are allowed and correspond to ignored pixels.

    For example, inputs of shape (nobj, ndim1, ndim2, n_data points)
    will give an output of shape (nobj, ndim1, ndim2).

    Parameters
    ----------
    dx, x_invvar : ndarray (..., ndim)

    Returns
    -------
    log of gaussian likelihood function : ndarray (...)

    """
    chi2s = np.sum(x_invvar * dx ** 2, axis=(-1))
    logs = np.where(x_invvar == 0, 0, np.log(x_invvar / 2 / np.pi))
    logdets = np.sum(logs, axis=(-1))
    return -0.5 * chi2s + 0.5 * logdets
