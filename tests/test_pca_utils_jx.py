from gasp.pca_utils_jx import *

import pytest

import jax
import jax.numpy as np

from chex import assert_shape

key = jax.random.PRNGKey(42)


def test_take_batch():

    npix = 10
    npix_tot = 10
    nobj = 22
    n_components = 3

    pcacomponents = jax.random.normal(key, (n_components, npix_tot))

    start_indices = jax.random.randint(key, (nobj,), minval=0, maxval=npix_tot - npix)

    pcacomponents_atz = take_batch(pcacomponents, start_indices, npix)

    assert_shape(pcacomponents_atz, (nobj, n_components, npix))
