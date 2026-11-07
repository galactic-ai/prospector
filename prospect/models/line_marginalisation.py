"""Marginalise linear line amplitudes under a uniform prior on a box.

Each amplitude a_j has the prior U(0, b_j), with b_j = W_j c_j for a uniform prior on the
equivalent width (W_j the limit, c_j the continuum that turns EW into flux). Following Johnson et
al. (2021, appendix E), the likelihood is normal in the amplitudes about the MLE a_hat with
covariance Sigma, so the integrated likelihood is

    ln L(theta) = ln L(theta, a_hat) + 1/2 ln|2 pi Sigma| - sum_j ln b_j + ln P_B,

with P_B the probability of the box under N(a_hat, Sigma). The derivation is in UberSED
knowledge/free_line_prior_derivation_2026-11-06.md. With independent pixel noise, two amplitudes
are correlated only if their templates share fitted pixels, so P_B is a product over connected
blocks of columns. A block of 1 uses the normal CDF, a block of 2 a fixed Gauss-Legendre rule and a
block of 3 or more quasi Monte Carlo with fixed seeds, both on Genz's separation of variables.
"""

import warnings
from collections import namedtuple
from functools import lru_cache

import numpy as np
from scipy import special
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import qmc

__all__ = ["LineColumn", "LineMarginal", "mle_amplitudes", "marginalise", "box_lnprob",
           "truncated_mean", "connected_blocks"]

# Gauss-Legendre nodes for a block of 2; ln P converges slowest for positive correlation deep in
# the tails, to 6e-5 at 20 sigma below zero.
GL_NODES = 512
# Quasi Monte Carlo for a block of 3 or more: independent scrambles, points per scramble (first
# and largest), and the target standard error of ln P.
QMC_SEEDS = tuple(range(8))
QMC_POINTS = (2**8, 2**14)
QMC_LN_TOL = 0.01

LineColumn = namedtuple("LineColumn", ["name", "lines", "ew_max", "sigma", "velocity", "replace"],
                        defaults=("eline_sigma", None, True))
LineColumn.__doc__ = """One free line amplitude, shared by a group of lines with the SPS line ratios.

name : label of the column.
lines : SPS line names in the column.
ew_max : upper limit W of the rest-frame equivalent width in Angstrom, summed over the lines.
sigma : name of the parameter that holds the intrinsic line width in km/s.
velocity : name of the parameter that holds a velocity offset in km/s, or None.
replace : if True the column replaces the SPS fluxes of its lines, otherwise it adds to them.
"""

LineMarginal = namedtuple("LineMarginal", ["amplitude", "covariance", "ln_penalty", "blocks"])
LineMarginal.__doc__ = """Result of :func:`marginalise`.

amplitude : MLE amplitudes, zero for columns without data.
covariance : covariance of the MLE amplitudes, zero rows for columns without data.
ln_penalty : 1/2 ln|2 pi Sigma| - sum ln b + ln P_B, to add to ln L(theta, a_hat).
blocks : column indices of each connected block, columns without data left out.
"""


def mle_amplitudes(resid, ivar, templates):
    """Return the MLE amplitudes, their covariance and ln det of their precision.

    Johnson et al. (2021, appendix E): a_hat = (G^T C^-1 G)^-1 G^T C^-1 resid and
    Sigma = (G^T C^-1 G)^-1, with C the data covariance.

    Parameters
    ----------
    resid : ndarray of shape (n,)
        Data minus the model without the amplitudes.
    ivar : ndarray of shape (n,) or (n, n)
        Inverse variance of each pixel (zero for pixels not fitted), or the inverse of the data
        covariance matrix.
    templates : ndarray of shape (n, k)
        Spectrum of each amplitude at unit amplitude.

    Returns
    -------
    amplitude : ndarray of shape (k,)
    covariance : ndarray of shape (k, k)
    ln_det_precision : float

    Raises
    ------
    numpy.linalg.LinAlgError
        If the precision G^T C^-1 G is not positive definite.
    """
    precision, rhs = _normal_equations(resid, ivar, templates)
    cov, ln_det = _invert(precision)
    return cov @ rhs, cov, ln_det


def marginalise(resid, ivar, templates, upper):
    """Marginalise the amplitudes of ``templates`` under uniform priors on ``[0, upper]``.

    Parameters
    ----------
    resid : ndarray of shape (n,)
        Data minus the model without the marginalised lines.
    ivar : ndarray of shape (n,) or (n, n)
        Inverse variance of each pixel (zero for pixels not fitted), or the inverse of the data
        covariance matrix.
    templates : ndarray of shape (n, k)
        Spectrum of each column at unit amplitude.
    upper : ndarray of shape (k,)
        Upper bound b of each amplitude.

    Returns
    -------
    LineMarginal
        The MLE amplitudes, their covariance, the log likelihood penalty and the blocks.

    Raises
    ------
    ValueError
        If a bound is not positive. ``numpy.linalg.LinAlgError`` (a ``ValueError``) if a block
        has no full rank.
    """
    upper = np.asarray(upper, dtype=float)
    if not np.all(upper > 0):
        raise ValueError("every amplitude bound must be positive")
    k = templates.shape[1]
    precision, rhs = _normal_equations(resid, ivar, templates)
    # a column without fitted pixels integrates to one over its prior and drops out
    fitted = np.flatnonzero(np.diag(precision) > 0)
    amplitude, covariance = np.zeros(k), np.zeros((k, k))
    blocks, ln_penalty = [], 0.0
    for sub in connected_blocks(precision[np.ix_(fitted, fitted)]):
        idx = fitted[sub]
        cov, ln_det_precision = _invert(precision[np.ix_(idx, idx)])
        amp = cov @ rhs[idx]
        amplitude[idx] = amp
        covariance[np.ix_(idx, idx)] = cov
        ln_penalty += 0.5 * (len(idx) * np.log(2 * np.pi) - ln_det_precision)
        ln_penalty += box_lnprob(amp, cov, upper[idx]) - np.sum(np.log(upper[idx]))
        blocks.append(idx)
    return LineMarginal(amplitude, covariance, float(ln_penalty), blocks)


def connected_blocks(precision):
    """Group columns into blocks connected by nonzero off-diagonal precision.

    Parameters
    ----------
    precision : ndarray of shape (k, k)
        Precision matrix of the amplitudes.

    Returns
    -------
    list of ndarray
        Column indices of each block, ordered by their first column.
    """
    if len(precision) == 0:
        return []
    _, label = connected_components(csr_matrix(precision != 0), directed=False)
    order = np.unique(label, return_index=True)[1]
    return [np.flatnonzero(label == label[i]) for i in np.sort(order)]


def box_lnprob(mean, cov, upper, lower=None):
    """Return ln P(lower <= a <= upper) for a ~ N(mean, cov).

    Parameters
    ----------
    mean : ndarray of shape (k,)
    cov : ndarray of shape (k, k)
    upper : ndarray of shape (k,)
    lower : ndarray of shape (k,), optional
        Zero by default.

    Returns
    -------
    float
        Exact for k = 1, a Gauss-Legendre rule for k = 2, and quasi Monte Carlo with fixed
        seeds for k >= 3, deterministic in every case.
    """
    mean = np.asarray(mean, dtype=float)
    lower = np.zeros_like(mean) if lower is None else np.asarray(lower, dtype=float)
    lo, hi = lower - mean, np.asarray(upper, dtype=float) - mean
    k = mean.size
    s = np.sqrt(np.diag(cov))
    ln_marginal = _log_ndtr_diff(lo / s, hi / s)
    if k == 1:
        return float(ln_marginal[0])
    # least probable variable first; P is unchanged, the integrand is smoother
    order = np.argsort(ln_marginal, kind="stable")
    lo, hi = lo[order], hi[order]
    chol = np.linalg.cholesky(cov[np.ix_(order, order)])
    if k == 2:
        x, w = _gauss_legendre(GL_NODES)
        lnf = _genz_lnintegrand(x[:, None], lo, hi, chol)
        return float(special.logsumexp(lnf, b=w))
    return _genz_qmc(lo, hi, chol)


def truncated_mean(amplitude, covariance, upper, blocks):
    """Return the mean of N(amplitude, covariance) truncated to the box [0, upper].

    This is the conditional posterior mean of the amplitudes under the box prior. For a block,
    integrating x phi(x) = -Sigma grad phi(x) by parts over the box gives
    E[x] = mu + Sigma F / P_B, where F_m is the density of x_m at its lower bound times the
    conditional box probability of the other amplitudes there, minus the same at its upper bound.

    Parameters
    ----------
    amplitude : ndarray of shape (k,)
    covariance : ndarray of shape (k, k)
    upper : ndarray of shape (k,)
    blocks : list of ndarray
        Column indices of each block, as returned by :func:`marginalise`.

    Returns
    -------
    ndarray of shape (k,)
        Truncated mean; columns in no block are returned unchanged.
    """
    out = np.array(amplitude, dtype=float)
    for idx in blocks:
        mu, cov, b = out[idx], covariance[np.ix_(idx, idx)], np.asarray(upper)[idx]
        ln_p = box_lnprob(mu, cov, b)
        f = np.zeros(len(idx))
        for m in range(len(idx)):
            s = np.sqrt(cov[m, m])
            rest = np.delete(np.arange(len(idx)), m)
            for edge, sign in ((0.0, 1.0), (b[m], -1.0)):
                t = edge - mu[m]
                term = -0.5 * (t / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi) - ln_p
                if len(rest):
                    gain = cov[rest, m] / cov[m, m]
                    ccov = cov[np.ix_(rest, rest)] - np.outer(gain, cov[m, rest])
                    term += box_lnprob(mu[rest] + gain * t, ccov, b[rest])
                f[m] += sign * np.exp(term)
        out[idx] = mu + cov @ f
    return out


def _normal_equations(resid, ivar, templates):
    """Return G^T C^-1 G and G^T C^-1 resid for a vector or matrix inverse covariance."""
    ivar = np.asarray(ivar)
    weighted = templates * ivar[:, None] if ivar.ndim == 1 else ivar @ templates
    return templates.T @ weighted, weighted.T @ resid


@lru_cache(maxsize=None)
def _gauss_legendre(n):
    """Return Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    x.flags.writeable = w.flags.writeable = False
    return x, w


@lru_cache(maxsize=None)
def _sobol(d, n, seed):
    """Return n scrambled Sobol points in [0, 1]^d for one seed."""
    points = qmc.Sobol(d, seed=seed).random(n)
    points.flags.writeable = False
    return points


def _invert(precision):
    """Return the inverse and ln det of a positive definite matrix, scaled to unit diagonal."""
    d = np.sqrt(np.diag(precision))
    chol = np.linalg.cholesky(precision / np.outer(d, d))
    inv = np.linalg.inv(chol)
    ln_det = 2 * np.sum(np.log(np.diag(chol))) + 2 * np.sum(np.log(d))
    return (inv.T @ inv) / np.outer(d, d), ln_det


def _log1mexp(x):
    """Return ln(1 - exp(x)) for x <= 0."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(x > -np.log(2), np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def _log_ndtr_diff(lo, hi):
    """Return ln(Phi(hi) - Phi(lo)) for lo <= hi, computed in the nearer tail."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    flip = lo + hi > 0
    a, b = np.where(flip, -hi, lo), np.where(flip, -lo, hi)
    lb = special.log_ndtr(b)
    return lb + _log1mexp(special.log_ndtr(a) - lb)


def _genz_lnintegrand(w, lo, hi, chol):
    """Return ln of Genz's integrand at points w in [0, 1]^(k-1), shape (npts, k-1).

    The box lo <= L z <= hi, with z standard normal and L lower triangular, is integrated one
    variable at a time: z_i is drawn inside its conditional interval, and the integrand is the
    product of the interval probabilities. Everything is done in logs, in the nearer tail.
    """
    k = len(lo)
    z = np.zeros((w.shape[0], k - 1))
    lnf = np.zeros(w.shape[0])
    for i in range(k):
        shift = z[:, :i] @ chol[i, :i]
        a, b = (lo[i] - shift) / chol[i, i], (hi[i] - shift) / chol[i, i]
        ln_e = _log_ndtr_diff(a, b)
        lnf += ln_e
        if i == k - 1:
            break
        with np.errstate(divide="ignore"):
            ln_w, ln_1mw = np.log(w[:, i]), np.log1p(-w[:, i])
        # z_i = Phi^-1(Phi(a) + w e) = -Phi^-1(Phi(-b) + (1 - w) e)
        flip = a + b > 0
        low = special.ndtri_exp(np.logaddexp(special.log_ndtr(a), ln_w + ln_e))
        high = -special.ndtri_exp(np.logaddexp(special.log_ndtr(-b), ln_1mw + ln_e))
        z[:, i] = np.where(flip, high, low)
    return lnf


def _genz_qmc(lo, hi, chol):
    """Return ln P of the box by scrambled Sobol points with fixed seeds.

    The points per scramble double until the standard error of ln P over the scrambles is below
    ``QMC_LN_TOL``; a warning is raised if the largest number of points does not reach it.
    """
    d = len(lo) - 1
    n = QMC_POINTS[0]
    while True:
        ln_p = np.array([
            special.logsumexp(_genz_lnintegrand(_sobol(d, n, s), lo, hi, chol)) - np.log(n)
            for s in QMC_SEEDS
        ])
        ln_mean = special.logsumexp(ln_p) - np.log(len(ln_p))
        ln_err = np.std(np.exp(ln_p - ln_mean), ddof=1) / np.sqrt(len(ln_p))
        if ln_err < QMC_LN_TOL or n >= QMC_POINTS[1]:
            break
        n *= 2
    if ln_err >= QMC_LN_TOL:
        warnings.warn(f"box probability error in ln P is {ln_err:.3g}, above {QMC_LN_TOL}", stacklevel=2)
    return float(ln_mean)
