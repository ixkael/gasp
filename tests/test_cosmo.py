from gasp.cosmo import D_L
from astropy.cosmology import FlatLambdaCDM
import pytest
import numpy as np


def test_approx_DL():

    for z in np.linspace(0.01, 4, num=10):
        v1 = D_L(z)
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None)
        v2 = cosmo.luminosity_distance(z).value
        assert abs(v1 / v2 - 1) < 0.01
