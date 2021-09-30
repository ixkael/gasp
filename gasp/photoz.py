# -*- coding: utf-8 -*-

import numpy as np
from gasp.cosmo import D_L
import scipy.interpolate
from sedpy import observate

from jax.scipy.special import gammaln


def igm_madau_tau(lam, zz):
    # ADAPTED FROM FSPS
    nspec = lam.size
    tau = np.zeros_like(lam)
    xc = np.zeros_like(lam)
    lobs = np.zeros_like(lam)
    igm_absorb = np.zeros_like(lam)
    #routine to include IGM absorption via Madau (1995)
    #this routine includes a fudge factor (accessed by pset%igm_factor)
    #that allows the user to scale the IGM optical depth

    nly = 17
    lyw = np.zeros((nly, ))
    lycoeff = np.zeros((nly, ))


    lyw = [1215.67, 1025.72, 972.537, 949.743, 937.803,
          930.748, 926.226, 923.150, 920.963, 919.352,
          918.129, 917.181, 916.429, 915.824, 915.329,
          914.919, 914.576]

    lycoeff = [0.0036,0.0017,0.0011846,0.0009410,0.0007960,
       0.0006967,0.0006236,0.0005665,0.0005200,0.0004817,
       0.0004487,0.0004200,0.0003947,0.000372,0.000352,
       0.0003334,0.00031644]

    lylim   = 911.75
    a_metal = 0.0017

    z1   = 1 + zz
    lobs = lam * z1
    xc   = lobs / lylim

    #Ly series line blanketing
    for i in range(nly):
        if lam[0] > lyw[i] or lam[-1] < lyw[i]:
            continue
        #vv = min(max(locate(lam, lyw[i]), 1), nspec)
        print(lam[0], lam[-1], lyw[i])
        vv = np.where(np.logical_and(lam[1:] > lyw[i], lam[:-1] < lyw[i]))[0][0]
        #print(lyw[i], lam[0], lam[-1], lam[vv], lam[vv+1])
        tau[0:vv] = tau[0:vv] + lycoeff[i] *(lobs[0:vv]/lyw[i])**3.46
        #add metal blanketing (this has ~no effect)
        if i == 1:
            tau[0:vv] = tau[0:vv] + a_metal*(lobs[0:vv]/lyw[i])**1.68

    #LyC absorption
    if lam[0] < lylim:
        #vv = min(max(locate(lam,lylim),1),nspec)
        vv = np.where(np.logical_and(lam[1:] > lylim, lam[:-1] < lylim))[0][0]
        #approximation to Eqn 16 in Madau (1995); see his footnote 3
        tau[0:vv] = tau[0:vv] +\
              (0.25*xc[0:vv]**3*(z1**0.46-xc[0:vv]**0.46)) +\
              (9.4*xc[0:vv]**1.5*(z1**0.18-xc[0:vv]**0.18)) -\
              (0.7*xc[0:vv]**3*(xc[0:vv]**(-1.32)-z1**(-1.32))) -\
              (0.023*(z1**1.68-xc[0:vv]**1.68))

    #the LyC fitting function seems to fall apart at really short
    #wavelengths, so when tau starts to decrease, cap it at the max.
    vv = np.argmax(tau)
    tau[0:vv] = tau[vv]

    #attenuate the input spectrum by the IGM
    #include a fudge factor to dial up/down the strength
    return tau


def interp(xnew, x, y):
    return scipy.interpolate.interp1d(
        x, y, kind="nearest", bounds_error=False, fill_value=0, assume_sorted=True
    )(xnew)


def logredshiftprior(x, a, b):
    # Eq 7 in https://arxiv.org/pdf/1807.01391.pdf
    logconst = (a + 1) / a * np.log(b) - np.log(a) + gammaln((a + 1) / a)
    val = a * np.log(x) - x ** a / b - logconst
    return val


def load_test_sed():
    """
    Loading test SED

    Parameters
    ----------
    None

    Returns
    -------
    lambda_aa, f_lambda_aa: ndarray (size, )
        arrays containing the wavelength (in Angstrom)
        and some rest-frame spectral energy distribution f_nu(lambda)

    """
    data = np.genfromtxt("data/seds/CWW/El_B2004a.dat")
    lambda_aa, f_lambda_aa = data[1:, 0], data[1:, 1]
    f_lambda_aa /= np.interp(3.5e3, lambda_aa, f_lambda_aa)

    return lambda_aa, f_lambda_aa


class PhotometricFilter:
    """
    Photometric filter response
    """

    def __init__(self, bandName, tabulatedWavelength, tabulatedResponse):

        self.bandName = bandName
        self.wavelength = tabulatedWavelength
        self.transmission = tabulatedResponse
        self.interp = interp1d(tabulatedWavelength, tabulatedResponse)
        self.norm = np.trapz(
            tabulatedResponse / tabulatedWavelength, x=tabulatedWavelength
        )
        ind = np.where(tabulatedResponse > 0.001 * np.max(tabulatedResponse))[0]
        self.lambdaMin = tabulatedWavelength[ind[0]]
        self.lambdaMax = tabulatedWavelength[ind[-1]]


def get_redshifted_photometry(lambda_aa, f_lambda_aa, redshift_grid, filter_list, apply_madau_igm=False):
    """
    -

    Parameters
    ----------
    lambda_aa, f_lambda_aa: ndarray (size, )
        wavelength (in Angstrom) and rest-frame spectral energy distribution f_nu(lambda)
    redshift_grid: ndarray (size, )
        array of redshift values to compute the photometry on
    filter_list: list of strings
        names of the photometric filters (will load with the SEDPY package)

    Returns
    -------

    https://www.roe.ac.uk/ifa/postgrad/pedagogy/2008_phillips.pdf
    https://arxiv.org/pdf/astro-ph/0210394.pdf

    """

    numBands = len(filter_list)
    redshift_factors = np.zeros((redshift_grid.size,))
    redshifted_fluxes = np.zeros((redshift_grid.size, numBands))
    redshifted_fluxes2 = np.zeros((redshift_grid.size, numBands))

    for iz, redshift in enumerate(redshift_grid):

        lambda_aa_redshifted = lambda_aa * (1 + redshift)

        redshift_factors[iz] = (
            D_L(redshift) ** 2.0 * (4 * np.pi) * (1 + redshift)
        ) ** -1
        # separate redshift factor
        f_lambda_aa_redshifted = f_lambda_aa
        f_nu_aa_redshifted = f_lambda_aa_redshifted * lambda_aa_redshifted ** 2 / 3e18

        if apply_madau_igm:
            tau = igm_madau_tau(lambda_aa_redshifted, redshift)
            f_nu_aa_redshifted *= np.exp(-tau)

        # get magnitudes using sedpy
        mags = observate.getSED(
            lambda_aa_redshifted, f_lambda_aa_redshifted, filterlist=filter_list
        )
        redshifted_fluxes[iz, :] = 10 ** (
            -0.4 * (mags + 48.60)
        )  # need to convert back from AB to cgs

        for ib, filter in enumerate(filter_list):

            filt_lambda, filt_spec = filter.wavelength * 1, filter.transmission * 1

            # need to normalize by this integral to satisfy photometry equations
            filt_spec /= np.trapz(filt_spec / filt_lambda, x=filt_lambda)

            # interpolate filter on redshifted lambda_aa_redshifted grid
            tspec = interp(lambda_aa_redshifted, filt_lambda, filt_spec)
            # print(lambda_aa_redshifted)
            redshifted_fluxes2[iz, ib] = np.trapz(
                f_nu_aa_redshifted * tspec / lambda_aa_redshifted,
                x=lambda_aa_redshifted,
            )

            # tODO: implement the other way around.
            # tspec = interp(filt_lambda, spec_lambda * (1 + zred), spec_f_lambda)
            # redshifted_fluxes3[iz, ib] = np.trapz(tspec * filt_spec / filt_lambda, x=filt_lambda)

    return redshifted_fluxes, redshifted_fluxes2, redshift_factors


def build_restframe_photometric_transferfunction(
    redshift_grid, spec_lambda, filter_list, f_lambda=True
):
    """
    -

    Parameters
    ----------
    - : ndarray ()
        -

    Returns
    -------

    """

    spec_lambda_sizes = np.diff(spec_lambda)
    spec_lambda_sizes = np.concatenate(([spec_lambda_sizes[0]], spec_lambda_sizes))

    numBands = len(filter_list)
    transfer_functions = np.zeros((redshift_grid.size, spec_lambda.size, numBands))
    redshift_factors = np.zeros((redshift_grid.size,))

    for iz, redshift in enumerate(redshift_grid):

        redshift_factors[iz] = (
            D_L(redshift) ** 2.0 * (4 * np.pi) * (1 + redshift)
        ) ** -1

        for ib, filt in enumerate(filter_list):
            filt_lambda, filt_spec = filt.wavelength * 1, filt.transmission * 1
            filt_spec /= np.trapz(filt_spec / filt_lambda, x=filt_lambda)

            if f_lambda:
                factor = (1 + redshift) ** 2 * spec_lambda_sizes * spec_lambda / 3e18
                transfer_functions[iz, :, ib] = factor * interp(
                    spec_lambda, filt_lambda / (1 + redshift), filt_spec
                )
            else:
                factor = (1 + redshift) ** 2 * spec_lambda_sizes / spec_lambda
                transfer_functions[iz, :, ib] = factor * interp(
                    spec_lambda, filt_lambda / (1 + redshift), filt_spec
                )

    return transfer_functions, redshift_factors
