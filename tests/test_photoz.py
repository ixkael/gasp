# -*- coding: utf-8 -*-

from gasp.photoz import *
import pytest
import numpy as np
from sedpy import observate


def test_photometry_and_transferfunctions():

    filternames = ["decam_g", "decam_r", "decam_z"]
    filterdir = "data/filters"
    filter_list = observate.load_filters(filternames, directory=filterdir)

    lambda_aa, f_lambda_aa = load_test_sed()
    speedoflight = 3e18
    f_nu_aa = f_lambda_aa * lambda_aa ** 2 / speedoflight

    redshift_grid = np.linspace(0.0, 2, 2)[1:]

    redshifted_fluxes, redshifted_fluxes2, redshift_factor = get_redshifted_photometry(
        lambda_aa, f_lambda_aa, redshift_grid, filter_list
    )

    relative_accuracy = 0.01
    print(redshifted_fluxes)
    assert np.allclose(redshifted_fluxes, redshifted_fluxes2, rtol=relative_accuracy)

    (
        transfer_functions_f_lambda,
        redshift_factor4,
    ) = build_restframe_photometric_transferfunction(
        redshift_grid, lambda_aa, filter_list, f_lambda=True
    )

    (
        transfer_functions_f_nu,
        redshift_factor5,
    ) = build_restframe_photometric_transferfunction(
        redshift_grid, lambda_aa, filter_list, f_lambda=False
    )

    redshifted_fluxes4 = np.sum(
        transfer_functions_f_lambda * f_lambda_aa[None, :, None], axis=1
    )
    redshifted_fluxes5 = np.sum(
        transfer_functions_f_nu * f_nu_aa[None, :, None], axis=1
    )

    rtol, atol = 1e-5, 1e-8
    assert np.allclose(redshifted_fluxes, redshifted_fluxes4, rtol=rtol, atol=atol)
    assert np.allclose(redshift_factor, redshift_factor4, rtol=rtol, atol=atol)
    assert np.allclose(redshifted_fluxes, redshifted_fluxes5, rtol=rtol, atol=atol)
    assert np.allclose(redshift_factor, redshift_factor5, rtol=rtol, atol=atol)
