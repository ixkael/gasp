from gasp.marginallikelihoods_jx import *
from gasp.stats import *

import pytest

import numpy as onp

import jax.random
import jax.numpy as np
from jax import grad, jit, vmap, hessian
from jax.scipy.special import logsumexp

from chex import assert_shape

key = jax.random.PRNGKey(42)

relative_accuracy = 0.001


def assert_fml_thetamap_thetacov(
    logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
):
    """
    Assert that three pairs of numpy arrays are close to each other numerically.
    """

    assert np.allclose(logfml, logfml2, rtol=relative_accuracy)
    assert np.allclose(theta_map, theta_map2, rtol=relative_accuracy)
    assert np.allclose(theta_cov, theta_cov2, rtol=relative_accuracy)


def design_matrix_polynomials(n_components, npix):
    """
    Generate a totally arbitrary matrix of polymials.

    Returns matrix of shape (n_components, npix)
    where the i-th row is the polynomial x ** i
    with x some arbitrary pixelisation, here of [-1, 1]
    """
    x = np.linspace(-1, 1, npix)
    return np.vstack([x ** i for i in np.arange(n_components)])


def test_logmarglike_lineargaussianmodel_onetransfer_batched():

    # shapes of arrays are in parentheses.

    # generate some fake data (noisy and masked)
    nobj = 14
    n_components = 2

    # true linear parameters, per object
    theta_truth = jax.random.normal(key, (nobj, n_components))

    n_pix_y = 100  # number of pixels for each object
    M_T = design_matrix_polynomials(n_components, n_pix_y)  # (n_components, n_pix_y)
    y_truth = np.matmul(theta_truth, M_T)  # (nobj, n_pix_y)
    y, yinvvar, logyinvvar = make_masked_noisy_data(y_truth)  # (nobj, n_pix_y)
    assert_equal_shape([y_truth, y, yinvvar, logyinvvar])
    # importantly, yinvvar and logyinvvar has zeros, symbolising ignored/masked pixels

    # now given the data and model matrix M_T,
    # compute the evidences, best fit thetas, and their covariances, for all objects at once.
    (
        logfml,
        theta_map,
        theta_cov,
    ) = logmarglike_lineargaussianmodel_onetransfer_jitvmap(y, yinvvar, M_T, logyinvvar)

    # checking shapes of output arrays
    assert_shape(logfml, (nobj,))
    assert_shape(theta_map, (nobj, n_components))
    assert_shape(theta_cov, (nobj, n_components, n_components))

    # add more tets
    # run optimisation of design matrix giv
    def loss_fn(M_T_new):
        (
            logfml,
            theta_map,
            theta_cov,
        ) = logmarglike_lineargaussianmodel_onetransfer_jitvmap(
            y, yinvvar, M_T_new, logyinvvar
        )
        return -np.sum(logfml)

    M_T_new_initial = jax.random.normal(key, (n_components, n_pix_y))
    param_list = [1 * M_T_new_initial]
    num_iterations = 10
    learning_rate = 1e-5
    # TODO: use better optimizer
    for n in range(num_iterations):
        grad_list = grad(loss_fn)(*param_list)
        param_list = [
            param - learning_rate * grad for param, grad in zip(param_list, grad_list)
        ]

    # optimised matrix:
    M_T_new_optimised = param_list[0]


def test_logmarglike_lineargaussianmodel_onetransfer():

    n_components = 2
    theta_truth = jax.random.normal(key, (n_components,))

    n_pix_y = 100
    M_T = design_matrix_polynomials(n_components, n_pix_y)
    yinvvar_ = 10 ** jax.random.normal(key, (n_pix_y,))
    y_truth = np.matmul(np.transpose(M_T), theta_truth)
    y_ = y_truth + jax.random.normal(key, (n_pix_y,)) * (yinvvar_ ** -0.5)
    mask_y = onp.random.uniform(size=n_pix_y) < 0.5
    yinvvar = yinvvar_ * mask_y
    y = y_ * mask_y
    logyinvvar = np.where(yinvvar == 0, 0, np.log(yinvvar))

    # first run
    logfml, theta_map, theta_cov = logmarglike_lineargaussianmodel_onetransfer(
        y_, yinvvar_, M_T
    )

    # check result is finite and shapes are correct
    assert np.isfinite(logfml)
    assert theta_map.shape[0] == n_components
    # assert theta_map.shape[1] == 1
    assert theta_cov.shape[0] == n_components
    assert theta_cov.shape[1] == n_components

    dt = theta_map - theta_truth
    chi2 = 0.5 * np.ravel(np.matmul(dt.T, np.linalg.solve(theta_cov, dt)))
    assert chi2 < 100

    # now run with some zeros
    logfml, theta_map, theta_cov = logmarglike_lineargaussianmodel_onetransfer(
        y_, yinvvar, M_T
    )
    # should get same answer by setting data to zero at same pixels
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_onetransfer(
        y, yinvvar, M_T
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
    )

    # check that posterior distribution is equal to product of gaussians too
    def logprob(theta):
        y_mod = np.matmul(theta, M_T)  # (n_samples, n_pix_y)
        return batch_gaussian_loglikelihood(y_mod - y, yinvvar)

    def logprob2(theta):
        dt = theta - theta_map
        s, logdet = np.linalg.slogdet(theta_cov * 2 * np.pi)
        chi2 = np.dot(dt.T, np.linalg.solve(theta_cov, dt))
        return logfml - 0.5 * (s * logdet + chi2)

    logpostv = logprob(theta_truth)
    logpostv2 = logprob2(theta_truth)
    assert abs(logpostv2 / logpostv - 1) < 0.01

    # now trying jit
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_onetransfer_jit(
        y, yinvvar, M_T, logyinvvar
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
    )

    def loss_fn(theta):
        return -logprob(theta)

    params = [1 * theta_map]
    learning_rate = 1e-5
    for n in range(10):
        grads = grad(loss_fn)(*params)
        params = [param - learning_rate * grad for param, grad in zip(params, grads)]
        # print(n, loss_fn(*params), params[0] - theta_map)
    assert np.allclose(theta_map, params[0], rtol=1e-6)

    # Testing analytic covariance is correct
    theta_cov2 = np.linalg.inv(np.reshape(hessian(loss_fn)(theta_map), theta_cov.shape))
    assert np.allclose(theta_cov, theta_cov2, rtol=1e-6)

    loss_fn_vmap = jit(vmap(loss_fn))

    n = 15
    theta_std = np.diag(theta_cov) ** 0.5
    theta_samples, vol_element = generate_sample_grid(theta_map, theta_std, n)
    loglikelihoods = -loss_fn_vmap(theta_samples)
    logfml_numerical = logsumexp(np.log(vol_element) + loglikelihoods)
    # print("logfml, logfml_numerical", logfml, logfml_numerical)
    assert abs(logfml_numerical / logfml - 1) < 0.01

    # compare 1d and 2d with large variance
    mu = theta_map * 0
    muinvvar = 1 / (1e4 * np.diag(theta_cov) ** 0.5)
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_twotransfers(
        y, yinvvar, M_T, mu, muinvvar
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, 0.2
    )


def test_logmarglike_lineargaussianmodel_twotransfers():

    n_components = 2
    theta_truth = jax.random.normal(key, (n_components,))
    mu = theta_truth * 1.1
    muinvvar = (theta_truth) ** -2
    mucov = np.eye(n_components) / muinvvar
    logmuinvvar = np.where(muinvvar == 0, 0, np.log(muinvvar))

    n_pix_y = 100
    M_T = design_matrix_polynomials(n_components, n_pix_y)
    yinvvar_ = 10 ** jax.random.normal(key, (n_pix_y,))
    y_truth = np.matmul(np.transpose(M_T), theta_truth)
    y_ = y_truth + jax.random.normal(key, (n_pix_y,)) / yinvvar_
    mask_y = onp.random.uniform(size=n_pix_y) < 0.5
    yinvvar = yinvvar_ * mask_y
    y = y_ * mask_y
    logyinvvar = np.where(yinvvar == 0, 0, np.log(yinvvar))

    n_pix_z = 10
    R_T = design_matrix_polynomials(n_components, n_pix_z)
    zinvvar_ = 10 ** jax.random.normal(key, (n_pix_z,))
    z_truth = np.matmul(np.transpose(R_T), theta_truth)
    z_ = z_truth + jax.random.normal(key, (n_pix_z,)) / zinvvar_
    mask_z = onp.random.uniform(size=n_pix_z) < 0.5
    zinvvar = zinvvar_ * mask_z
    z = z_ * mask_z
    logzinvvar = np.where(zinvvar == 0, 0, np.log(zinvvar))

    # first run
    logfml, theta_map, theta_cov = logmarglike_lineargaussianmodel_twotransfers(
        y_, yinvvar_, M_T, mu, muinvvar
    )
    # check result is finite and shapes are correct
    assert np.isfinite(logfml)
    assert theta_map.shape[0] == n_components
    assert theta_cov.shape[0] == n_components
    assert theta_cov.shape[1] == n_components

    # now run with some zeros
    logfml, theta_map, theta_cov = logmarglike_lineargaussianmodel_twotransfers(
        y_, yinvvar, M_T, mu, muinvvar
    )
    # should get same answer by setting data to zero at same pixels
    y_withzeros = y * mask_y
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_twotransfers(
        y, yinvvar, M_T, mu, muinvvar
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
    )

    # now trying jit
    logyinvvar = np.where(yinvvar == 0, 0, np.log(yinvvar))
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_twotransfers_jit(
        y_withzeros, yinvvar, M_T, mu, muinvvar, logyinvvar, logmuinvvar
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
    )

    # check that posterior distribution is equal to product of gaussians too
    def logprob(theta):
        y_mod = np.matmul(theta, M_T)  # (n_samples, n_pix_y)
        like = batch_gaussian_loglikelihood(y_mod - y, yinvvar)
        prior = batch_gaussian_loglikelihood(theta - mu, muinvvar)
        return like + prior

    def logprob2(theta):
        dt = theta - theta_map
        s, logdet = np.linalg.slogdet(theta_cov * 2 * np.pi)
        chi2 = np.dot(dt.T, np.linalg.solve(theta_cov, dt))
        return logfml - 0.5 * (s * logdet + chi2)

    logpostv = logprob(theta_truth)
    logpostv2 = logprob2(theta_truth)
    assert abs(logpostv2 / logpostv - 1) < 0.01

    def loss_fn(theta):
        return -logprob(theta)

    params = [1 * theta_map]
    learning_rate = 1e-5
    for n in range(10):
        grads = grad(loss_fn)(*params)
        params = [param - learning_rate * grad for param, grad in zip(params, grads)]
        # print(n, loss_fn(*params), params[0] - theta_map)
    assert np.allclose(theta_map, params[0], rtol=1e-6)

    # Testing analytic covariance is correct
    theta_cov2 = np.linalg.inv(np.reshape(hessian(loss_fn)(theta_map), theta_cov.shape))
    assert np.allclose(theta_cov, theta_cov2, rtol=1e-6)

    loss_fn_vmap = jit(vmap(loss_fn))

    n = 15
    theta_std = np.diag(theta_cov) ** 0.5
    theta_samples, vol_element = generate_sample_grid(theta_map, theta_std, n)
    logpost = -loss_fn_vmap(theta_samples)
    logfml_numerical = logsumexp(np.log(vol_element) + logpost)
    # print("logfml, logfml_numerical", logfml, logfml_numerical)
    assert abs(logfml_numerical / logfml - 1) < 0.01
