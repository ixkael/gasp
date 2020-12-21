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
