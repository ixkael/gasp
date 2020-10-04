# -*- coding: utf-8 -*-

import numpy as np


def D_L(z):
    """
    Approximation of luminosity distance.
    Agrees with astropy.FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None) well


    Parameters
    ----------
    z: ndarray (...)
        Redshift

    Returns
    -------
    ndarray (...)

    """
    return np.exp(30.5 * z ** 0.04 - 21.7)  # / 0.25e5
