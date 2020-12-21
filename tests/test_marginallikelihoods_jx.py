from gasp.marginallikelihoods_jx import *
from gasp.stats import *

import pytest

import numpy as onp

import jax.random
import jax.numpy as np
from jax import grad, jit, vmap, hessian
from jax.scipy.special import logsumexp

from chex import assert_shape, assert_equal_shape

key = jax.random.PRNGKey(42)

relative_accuracy = 0.001


def generate_sample_grid(theta_mean, theta_std, n):
    """
    Create a meshgrid of n ** n_dim samples,
    tiling [theta_mean[i] - 5 * theta_std[i], theta_mean[i] + 5 * theta_std]
    into n portions.
    Also returns the volume element.

    Parameters
    ----------
    theta_mean, theta_std : ndarray (n_dim)

    Returns
    -------
    theta_samples : ndarray (nobj, n_dim)

    vol_element: scalar
        Volume element

    """
    n_components = theta_mean.size
    xs = [
        np.linspace(
            theta_mean[i] - 5 * theta_std[i],
            theta_mean[i] + 5 * theta_std[i],
            n,
        )
        for i in range(n_components)
    ]
    mxs = np.meshgrid(*xs)
    orshape = mxs[0].shape
    mxsf = np.vstack([i.ravel() for i in mxs]).T
    dxs = np.vstack([np.diff(xs[i])[i] for i in range(n_components)])
    vol_element = np.prod(dxs)
    theta_samples = np.vstack(mxsf)
    return theta_samples, vol_element


def make_masked_noisy_data(y_truth, masked_fraction=0.5):
    """
    From some noiseless data, generates some arbitrary noise and mask.

    Parameters
    ----------
    y_truth : ndarray (nobj, npix)
        Input noiseless data

    masked_fraction: scalar
        Fraction of data to mask (default: 0.5)

    Returns
    -------
    y_withzeros, yinvvar_withzeros, logyinvvar_withzeros : ndarray (..., npix)
        output noisy data, inverse variance, and log inverse variance,
        with masked pixels (= zeros in yinvvar_withzeros and logyinvvar_withzeros)

    """
    if len(y_truth.shape) == 1:
        y_truth = y_truth[None, :]
    nobj, n_pix_y = y_truth.shape
    rn = jax.random.normal(key, (nobj, n_pix_y))
    yinvvar = 10 ** rn
    rn = jax.random.normal(key, (nobj, n_pix_y))
    y = y_truth + rn * (yinvvar ** -0.5)
    rn = onp.random.uniform(size=n_pix_y * nobj).reshape((nobj, n_pix_y))
    mask_y = rn < masked_fraction
    yinvvar_withzeros = yinvvar * mask_y
    y_withzeros = y * mask_y
    logyinvvar_withzeros = np.where(
        yinvvar_withzeros == 0, 0, np.log(yinvvar_withzeros)
    )
    return (
        y_withzeros.squeeze(),
        yinvvar_withzeros.squeeze(),
        logyinvvar_withzeros.squeeze(),
    )


def batch_gaussian_loglikelihood(dx, x_invvar):
    """
    log of gaussian likelihood function for a batch of data.
    IMPORTANTLY, zeros in x_invvar are allowed and correspond to ignored pixels.

    For example, inputs of shape (nobj, ndim1, ndim2, n_data points)
    will give an output of shape (nobj, ndim1, ndim2).

    Parameters
    ----------
    dx, x_invvar : ndarray (..., ndim)

    Returns
    -------
    log of gaussian likelihood function : ndarray (...)

    """
    chi2s = np.sum(x_invvar * dx ** 2, axis=(-1))
    logs = np.where(x_invvar == 0, 0, np.log(x_invvar / 2 / np.pi))
    logdets = np.sum(logs, axis=(-1))
    return -0.5 * chi2s + 0.5 * logdets


def assert_fml_thetamap_thetacov(
    logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
):
    """
    Assert that three pairs of numpy arrays are close to each other numerically.
    """
    assert_equal_shape([logfml, logfml2])
    assert_equal_shape([theta_map, theta_map2])
    assert_equal_shape([theta_cov, theta_cov2])

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
    M_T = design_matrix_polynomials(n_components, n_pix_y)  # (n_components, n_pix_y)
    y_truth = np.matmul(theta_truth, M_T)  # (nobj, n_pix_y)
    y, yinvvar, logyinvvar = make_masked_noisy_data(y_truth)  # (nobj, n_pix_y)
    assert_equal_shape([y_truth, y, yinvvar, logyinvvar])

    logfml, theta_map, theta_cov = logmarglike_lineargaussianmodel_onetransfer(
        y, yinvvar, M_T
    )

    # check result is finite and shapes are correct
    assert_shape(theta_map, (n_components,))
    assert_shape(theta_cov, (n_components, n_components))
    assert np.isfinite(logfml)
    assert np.all(np.isfinite(theta_map))
    assert np.all(np.isfinite(theta_cov))

    # check that result isn't too far off the truth, in chi2 sense
    dt = theta_map - theta_truth
    chi2 = 0.5 * np.ravel(np.matmul(dt.T, np.linalg.solve(theta_cov, dt)))
    assert chi2 < 100

    # check that normalised posterior distribution factorises into product of gaussians
    def log_posterior(theta):
        y_mod = np.matmul(theta, M_T)  # (n_samples, n_pix_y)
        return batch_gaussian_loglikelihood(y_mod - y, yinvvar)

    def log_posterior2(theta):
        dt = theta - theta_map
        s, logdet = np.linalg.slogdet(theta_cov * 2 * np.pi)
        chi2 = np.dot(dt.T, np.linalg.solve(theta_cov, dt))
        return logfml - 0.5 * (s * logdet + chi2)

    logpostv = log_posterior(theta_truth)
    logpostv2 = log_posterior2(theta_truth)
    assert abs(logpostv2 / logpostv - 1) < 0.01

    # now trying jit version of function
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_onetransfer_jit(
        y, yinvvar, M_T, logyinvvar
    )
    # check that outputs match original implementation
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
    )

    # now running simple optimiser to check that result is indeed optimum
    def loss_fn(theta):
        return -log_posterior(theta)

    params = [1 * theta_map]
    learning_rate = 1e-5
    for n in range(10):
        grads = grad(loss_fn)(*params)
        params = [param - learning_rate * grad for param, grad in zip(params, grads)]
        # print(n, loss_fn(*params), params[0] - theta_map)
    assert np.allclose(theta_map, params[0], rtol=1e-6)

    # Testing analytic covariance is correct and equals inverse of hessian
    theta_cov2 = np.linalg.inv(np.reshape(hessian(loss_fn)(theta_map), theta_cov.shape))
    assert np.allclose(theta_cov, theta_cov2, rtol=1e-6)

    # create vectorised loss
    loss_fn_vmap = jit(vmap(loss_fn))

    # now computes the evidence numerically
    n = 15
    theta_std = np.diag(theta_cov) ** 0.5
    theta_samples, vol_element = generate_sample_grid(theta_map, theta_std, n)
    loglikelihoods = -loss_fn_vmap(theta_samples)
    logfml_numerical = logsumexp(np.log(vol_element) + loglikelihoods)
    # print("logfml, logfml_numerical", logfml, logfml_numerical)
    assert abs(logfml_numerical / logfml - 1) < 0.01

    # Compare with case including gaussian prior with large variance
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
    M_T = design_matrix_polynomials(n_components, n_pix_y)  # (n_components, n_pix_y)
    y_truth = np.matmul(theta_truth, M_T)  # (nobj, n_pix_y)
    y, yinvvar, logyinvvar = make_masked_noisy_data(y_truth)  # (nobj, n_pix_y)

    # first run
    logfml, theta_map, theta_cov = logmarglike_lineargaussianmodel_twotransfers(
        y, yinvvar, M_T, mu, muinvvar
    )
    # check result is finite and shapes are correct
    assert np.isfinite(logfml)
    assert_shape(theta_map, (n_components,))
    assert_shape(theta_cov, (n_components, n_components))

    # now trying jit
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_twotransfers_jit(
        y, yinvvar, M_T, mu, muinvvar, logyinvvar, logmuinvvar
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
    )

    # check that posterior distribution is equal to product of gaussians too
    def log_posterior(theta):
        y_mod = np.matmul(theta, M_T)  # (n_samples, n_pix_y)
        like = batch_gaussian_loglikelihood(y_mod - y, yinvvar)
        prior = batch_gaussian_loglikelihood(theta - mu, muinvvar)
        return like + prior

    def log_posterior2(theta):
        dt = theta - theta_map
        s, logdet = np.linalg.slogdet(theta_cov * 2 * np.pi)
        chi2 = np.dot(dt.T, np.linalg.solve(theta_cov, dt))
        return logfml - 0.5 * (s * logdet + chi2)

    logpostv = log_posterior(theta_truth)
    logpostv2 = log_posterior2(theta_truth)
    assert abs(logpostv2 / logpostv - 1) < 0.01

    def loss_fn(theta):
        return -log_posterior(theta)

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


def test_logmarglike_lineargaussianmodel_threetransfers():

    n_components = 2
    theta_truth = jax.random.normal(key, (n_components,))
    mu = theta_truth * 1.1
    muinvvar = (theta_truth) ** -2
    mucov = np.eye(n_components) / muinvvar
    logmuinvvar = np.where(muinvvar == 0, 0, np.log(muinvvar))

    n_pix_y = 100
    M_T = design_matrix_polynomials(n_components, n_pix_y)  # (n_components, n_pix_y)
    y_truth = np.matmul(theta_truth, M_T)  # (nobj, n_pix_y)
    y, yinvvar, logyinvvar = make_masked_noisy_data(y_truth)  # (nobj, n_pix_y)

    ell = 1

    n_pix_z = 10
    R_T = design_matrix_polynomials(n_components, n_pix_z)
    z_truth = np.matmul(theta_truth, R_T)  # (nobj, n_pix_z)
    z, zinvvar, logzinvvar = make_masked_noisy_data(z_truth)  # (nobj, n_pix_z)

    # first run
    logfml, theta_map, theta_cov = logmarglike_lineargaussianmodel_threetransfers(
        ell, y, yinvvar, M_T, z, zinvvar, R_T, mu, muinvvar
    )
    # check result is finite and shapes are correct
    assert np.isfinite(logfml)
    print(logfml)
    assert_shape(theta_map, (n_components,))
    assert_shape(theta_cov, (n_components, n_components))

    # now trying jit
    (
        logfml2,
        theta_map2,
        theta_cov2,
    ) = logmarglike_lineargaussianmodel_threetransfers_jit(
        ell,
        y,
        yinvvar,
        M_T,
        z,
        zinvvar,
        R_T,
        mu,
        muinvvar,
        logyinvvar,
        logzinvvar,
        logmuinvvar,
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
    )

    # check that posterior distribution is equal to product of gaussians too
    def log_posterior(theta):
        y_mod = np.matmul(theta, M_T)  # (n_samples, n_pix_y)
        like_y = batch_gaussian_loglikelihood(y_mod - y, yinvvar)
        z_mod = ell * np.matmul(theta, R_T)  # (n_samples, n_pix_y)
        like_z = batch_gaussian_loglikelihood(z_mod - z, zinvvar)
        prior = batch_gaussian_loglikelihood(theta - mu, muinvvar)
        return like_y + like_z + prior

    def log_posterior2(theta):
        dt = theta - theta_map
        s, logdet = np.linalg.slogdet(theta_cov * 2 * np.pi)
        chi2 = np.dot(dt.T, np.linalg.solve(theta_cov, dt))
        return logfml - 0.5 * (s * logdet + chi2)

    logpostv = log_posterior(theta_truth)
    logpostv2 = log_posterior2(theta_truth)
    assert abs(logpostv2 / logpostv - 1) < 0.01

    def loss_fn(theta):
        return -log_posterior(theta)

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


def test_logmarglike_lineargaussianmodel_threetransfers():

    nobj = 10

    n_components = 2
    theta_truth = jax.random.normal(key, (nobj, n_components))
    mu = theta_truth * 1.1
    muinvvar = (theta_truth) ** -2
    mucov = np.eye(n_components) / muinvvar
    logmuinvvar = np.where(muinvvar == 0, 0, np.log(muinvvar))

    n_pix_y = 100
    M_T = design_matrix_polynomials(n_components, n_pix_y)  # (n_components, n_pix_y)
    y_truth = np.matmul(theta_truth, M_T)  # (nobj, n_pix_y)
    y, yinvvar, logyinvvar = make_masked_noisy_data(y_truth)  # (nobj, n_pix_y)

    n_pix_z = 10
    ells = 10 ** jax.random.normal(key, (nobj,))
    R_T = design_matrix_polynomials(n_components, n_pix_z)
    z_truth = ells * np.matmul(theta_truth, R_T)  # (nobj, n_pix_z)
    z, zinvvar, logzinvvar = make_masked_noisy_data(z_truth)  # (nobj, n_pix_z)

    # first run
    (
        logfml,
        theta_map,
        theta_cov,
    ) = logmarglike_lineargaussianmodel_threetransfers_jitvmap(
        ells,
        y,
        yinvvar,
        M_T,
        z,
        zinvvar,
        R_T,
        mu,
        muinvvar,
        logyinvvar,
        logzinvvar,
        logmuinvvar,
    )
    # check result is finite and shapes are correct
    assert np.isfinite(logfml)
    print(logfml)
    assert_shape(theta_map, (n_components,))
    assert_shape(theta_cov, (n_components, n_components))

    # check that posterior distribution is equal to product of gaussians too
    def log_posterior(theta):
        y_mod = np.matmul(theta, M_T)  # (n_samples, n_pix_y)
        like_y = batch_gaussian_loglikelihood(y_mod - y, yinvvar)
        z_mod = ell * np.matmul(theta, R_T)  # (n_samples, n_pix_y)
        like_z = batch_gaussian_loglikelihood(z_mod - z, zinvvar)
        prior = batch_gaussian_loglikelihood(theta - mu, muinvvar)
        return like_y + like_z + prior

    def log_posterior2(theta):
        dt = theta - theta_map
        s, logdet = np.linalg.slogdet(theta_cov * 2 * np.pi)
        chi2 = np.dot(dt.T, np.linalg.solve(theta_cov, dt))
        return logfml - 0.5 * (s * logdet + chi2)

    logpostv = log_posterior(theta_truth)
    logpostv2 = log_posterior2(theta_truth)
    assert abs(logpostv2 / logpostv - 1) < 0.01

    def loss_fn(theta):
        return -log_posterior(theta)

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
