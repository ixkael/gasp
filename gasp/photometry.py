# -*- coding: utf-8 -*-

import numpy as np


def flux2mag(fluxes, mag_zeropoint=0):
    """
    Flux to magnitude convertion
    """
    return -2.5 * np.log10(fluxes) - mag_zeropoint


def mag2flux(mags, mag_zeropoint=0):
    """
    Magnitude to flux correction
    """
    return 10 ** (-0.4 * (mags + mag_zeropoint))


def mag2flux_witherr(mags, magerrs, mag_zeropoint=0):
    """
    Magnitude to flux correction, with errors
    """
    fluxes = 10 ** (-0.4 * (mags + mag_zeropoint))
    fluxerrs = fluxes * (10 ** np.abs(0.4 * magerrs) - 1)
    return fluxes, fluxerrs
