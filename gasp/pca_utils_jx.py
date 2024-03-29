# -*- coding: utf-8 -*-

from functools import partial
from jax import jit
import jax.numpy as np
from jax.lax import dynamic_slice


@partial(jit, static_argnums=(2))
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
        (nobj, n_components, npix), dtype=int
    )
    indices_1 = indices_2d[:, None, :] * np.ones((nobj, n_components, npix), dtype=int)

    pcacomponents_atz = pcacomponents[indices_0, indices_1]

    return pcacomponents_atz
