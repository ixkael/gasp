# -*- coding: utf-8 -*-

from gasp.photoz import *
import pytest
import numpy as np
from sedpy import observate
import jax.random

key = jax.random.PRNGKey(42)


def test_logredshiftprior():

    for _ in range(10):
        x = np.linspace(0, 5, 100000)
        a = 10 ** (1 + jax.random.normal(key, (4,)) / 2)
        b = 10 ** (1 + jax.random.normal(key, (4,)) / 2)
        y = np.exp(logredshiftprior(x[:, None], a[None, :], b[None, :]))
        norms = np.trapz(y, x, axis=0)
        assert np.allclose(norms, np.ones_like(norms), rtol=1e-1)


def test_madau():

    filternames = ["decam_u", "decam_g"]
    filterdir = "data/filters"
    filter_list = observate.load_filters(filternames, directory=filterdir)

    lambda_aa, f_lambda_aa = load_test_sed()
    speedoflight = 3e18
    f_nu_aa = f_lambda_aa * lambda_aa ** 2 / speedoflight

    redshift_grid = np.linspace(0.0, 4, 4)[1:]

    redshifted_fluxes1, redshifted_fluxesb1, redshift_factor = get_redshifted_photometry(
        lambda_aa, f_lambda_aa, redshift_grid, filter_list, apply_madau_igm=False
    )
    redshifted_fluxes2, redshifted_fluxesb2, redshift_factor = get_redshifted_photometry(
        lambda_aa, f_lambda_aa, redshift_grid, filter_list, apply_madau_igm=True
    )

    assert np.any(redshifted_fluxes1 != redshifted_fluxes2)
    assert np.any(redshifted_fluxesb1 != redshifted_fluxesb2)


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
