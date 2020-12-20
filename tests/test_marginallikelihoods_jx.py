from gasp.marginallikelihoods_jx import *
import pytest
import jax.random
import jax.numpy as np
import numpy as onp
from jax import grad, jit, vmap, hessian
from jax.scipy.special import logsumexp

key = jax.random.PRNGKey(42)

relative_accuracy = 0.001


def assert_fml_thetamap_thetacov(
    logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, relative_accuracy
):

    assert abs(logfml2 / logfml - 1) < relative_accuracy
    assert np.allclose(theta_map, theta_map2, rtol=relative_accuracy)
    assert np.allclose(theta_cov, theta_cov2, rtol=relative_accuracy)


def batch_transpose_2d(x):
    return np.transpose(x, [0, 2, 1])


def batch_gaussian_loglikelihood(dx, x_invvar):
    chi2s = np.sum(x_invvar * dx ** 2, axis=(-1))
    logs = np.where(x_invvar == 0, 0, np.log(x_invvar / 2 / np.pi))
    logdets = np.sum(logs, axis=(-1))
    return -0.5 * chi2s + 0.5 * logdets


def design_matrix_polynomials(n_components, npix):
    x = np.arange(npix)
    return np.vstack([x ** i for i in np.arange(n_components)])


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
    def logprob(
        theta,
    ):  # theta is (..., n_components)  M_T is (n_components, n_pix_y)
        y_mod = np.matmul(theta, M_T)  # (n_samples, n_pix_y, 1)
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

    # now numerically sample best fit parameters
    n_samples = 1000
    theta_samples = jax.random.multivariate_normal(
        key,
        theta_map,
        theta_cov,
        shape=(n_samples,),
    )  # n_samples, n_components
    y_mod = np.matmul(theta_samples, M_T)  # (n_samples, n_pix_y)
    loglikelihoods = batch_gaussian_loglikelihood(y_mod - y[None, :], yinvvar[None, :])
    logfml_numerical = logsumexp(loglikelihoods) - n_samples

    def loss_fn(theta):  # theta is (..., n_components)  M_T is (n_components, n_pix_y)
        y_mod = np.matmul(theta, M_T)  # (..., n_pix_y)
        # chi2s = np.sum(x_invvar * dx ** 2, axis=(-2, -1))
        # logs = np.where(x_invvar == 0, 0, np.log(x_invvar / 2 / np.pi))
        # logdets = np.sum(logs, axis=(-2, -1))
        return -np.sum(
            batch_gaussian_loglikelihood(y_mod - y[None, :], yinvvar[None, :])
        )

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
    theta_std = theta_cov ** 0.5
    xs = [
        np.linspace(
            theta_map2[i] - 5 * theta_std[i, i],
            theta_map2[i] + 5 * theta_std[i, i],
            n,
        )
        for i in range(n_components)
    ]
    mxs = np.meshgrid(*xs)
    orshape = mxs[0].shape
    mxsf = np.vstack([i.ravel() for i in mxs]).T
    theta_samples = np.vstack(mxsf)
    loglikelihoods = -loss_fn_vmap(theta_samples)
    dxs = np.vstack([np.diff(xs[i])[i] for i in range(n_components)])
    vol = np.prod(dxs)
    logfml_numerical = logsumexp(np.log(vol) + loglikelihoods)
    # print("logfml, logfml_numerical", logfml, logfml_numerical)
    assert abs(logfml_numerical / logfml - 1) < 0.01

    # compare 1d and 2d with large variance
    mu = theta_map * 0
    muinvvar = 1 / (1e4 * np.diag(theta_cov) ** 0.5)
    logfml2, theta_map2, theta_cov2 = logmarglike_lineargaussianmodel_twotransfers(
        y, yinvvar, M_T, mu, muinvvar
    )
    assert_fml_thetamap_thetacov(
        logfml, theta_map, theta_cov, logfml2, theta_map2, theta_cov2, 0.1
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

    # test jit and compare speed with precalculcated logs
    # test vectorization
