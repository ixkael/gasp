# -*- coding: utf-8 -*-

import jax.numpy as np


def take_batch(pcacomponents, start_indices, npix):
    """

    Parameters
    ----------
    pcacomponents: ndarray (n_components, large_nb_of_pixels)
        Redshift
    start_indices:  ndarray (nobj, )
        Indices
    npix: int
        Number of pixels to fetch

    Returns
    -------
    ndarray (nobj, n_components, npix)

    """
    n_components, specwavesize = pcacomponents.shape
    nobj = start_indices.shape[0]

    indices_2d = start_indices[:, None] + np.arange(npix)[None, :]

    indices_0 = np.arange(n_components)[None, :, None] * np.ones(
        (nobj, n_components, specwavesize), dtype=int
    )
    indices_1 = indices_2d[:, None, :] * np.ones(
        (nobj, n_components, specwavesize), dtype=int
    )

    pcacomponents_atz = pcacomponents[indices_0, indices_1]

    return pcacomponents_atz
