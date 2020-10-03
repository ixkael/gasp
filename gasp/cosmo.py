import numpy as np


def D_L(z):
    """
    Approximation of luminosity distance

    Parameters
    ----------
    z: ndarray (...)
        Redshift

    Returns
    -------
    ndarray (...)

    """
    return np.exp(30.5 * z ** 0.04 - 21.7) / 0.25e5
