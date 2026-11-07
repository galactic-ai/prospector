"""Line amplitudes marginalised under a uniform prior on a box, against brute-force integrals.

The toy model and its three cases are those of section 8 of the derivation note
(UberSED knowledge/free_line_prior_derivation_2026-11-06.md), whose committed values come from
hubersed experiments/2026-11-07_line_prior_check.
"""

import numpy as np
import pytest
from scipy import integrate, special, stats

from prospect.models import line_marginalisation as lm

X = np.linspace(-10.0, 10.0, 200)
SIGMA = np.full(X.size, 0.3)
IVAR = 1.0 / SIGMA**2
G = np.stack([np.exp(-0.5 * (X + 0.6) ** 2), np.exp(-0.5 * (X - 0.6) ** 2)], axis=1)
W = np.array([3.0, 3.0])
C = np.array([2.0, 2.5])
# true amplitudes and the committed brute-force ln L(theta) of each section 8 case (seed 1)
CASES = {
    "one strong, one weak line": ([0.3, 4.0], -35.89113),
    "both lines near zero": ([0.05, 0.02], -37.21138),
    "line 1 at its upper bound": ([6.0, 1.0], -36.43118),
}


def toy(truth, seed):
    """Return the toy data and its Gaussian log likelihood as a function of the amplitudes."""
    rng = np.random.default_rng(seed)
    data = G @ np.asarray(truth) + rng.normal(0.0, SIGMA)

    def lnlike(a):
        r = data - G @ a
        return -0.5 * np.sum(IVAR * r**2) - 0.5 * np.sum(np.log(2 * np.pi * SIGMA**2))

    return data, lnlike


def log_box_1d(lo, hi):
    """Return ln(Phi(hi) - Phi(lo)), stable in both tails."""
    if lo + hi > 0:
        lo, hi = -hi, -lo
    return special.log_ndtr(hi) + np.log1p(-np.exp(special.log_ndtr(lo) - special.log_ndtr(hi)))


def ref_box_2d(mean, cov, upper):
    """Return ln P of the box [0, upper] by adaptive quadrature over the first amplitude."""
    s1, s2 = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (s1 * s2)
    sc = s2 * np.sqrt(1 - rho**2)

    def lnf(x):
        m2 = mean[1] + rho * s2 * x
        return stats.norm.logpdf(x) + log_box_1d(-m2 / sc, (upper[1] - m2) / sc)

    lo, hi = -mean[0] / s1, (upper[0] - mean[0]) / s1
    grid = np.linspace(lo, hi, 2001)
    shift = max(lnf(x) for x in grid)
    val, _ = integrate.quad(
        lambda x: np.exp(lnf(x) - shift), lo, hi, epsabs=0, epsrel=1e-12, limit=500, points=grid[::200]
    )
    return np.log(val) + shift


# %% section 8: closed form against brute force


@pytest.mark.parametrize("name", CASES)
def test_section8_cases(name):
    """The log likelihood at a_hat plus the penalty equals the integral over the box, for each case."""
    truth, committed = CASES[name]
    data, lnlike = toy(truth, seed=1)
    b = W * C
    m = lm.marginalise(data, IVAR, G, b)
    val, _ = integrate.dblquad(
        lambda a2, a1: np.exp(lnlike(np.array([a1, a2])) - lnlike(m.amplitude)),
        0.0, b[0], 0.0, b[1], epsabs=1e-13, epsrel=1e-11,
    )  # fmt: skip
    brute = np.log(val) + lnlike(m.amplitude) - np.sum(np.log(b))
    closed = lnlike(m.amplitude) + m.ln_penalty
    assert closed == pytest.approx(brute, abs=1e-6)
    assert closed == pytest.approx(committed, abs=1e-5)


@pytest.mark.parametrize("c", [np.array([2.0, 2.5]), np.array([0.7, 4.0])])
def test_jacobian_ew_space(c):
    """Integrating over EW ~ U(0, W) with a = EW c gives the same ln L as the box in a."""
    data, lnlike = toy([0.3, 4.0], seed=2)
    m = lm.marginalise(data, IVAR, G, W * c)
    val, _ = integrate.dblquad(
        lambda e2, e1: np.exp(lnlike(np.array([e1, e2]) * c) - lnlike(m.amplitude)),
        0.0, W[0], 0.0, W[1], epsabs=1e-13, epsrel=1e-11,
    )  # fmt: skip
    direct = np.log(val) + lnlike(m.amplitude) - np.sum(np.log(W))
    assert lnlike(m.amplitude) + m.ln_penalty == pytest.approx(direct, abs=1e-6)


def test_mle_matches_normal_equations():
    """The amplitudes and covariance are the MLE and its covariance (Johnson et al. App. E)."""
    data, _ = toy([0.3, 4.0], seed=1)
    m = lm.marginalise(data, IVAR, G, W * C)
    cov = np.linalg.inv(G.T @ (G * IVAR[:, None]))
    assert np.allclose(m.covariance, cov, rtol=1e-12)
    assert np.allclose(m.amplitude, cov @ (G.T @ (IVAR * data)), rtol=1e-10)


def test_wide_box_is_flat_prior():
    """With every amplitude far inside the box, the penalty is App. E's flat term minus sum ln b."""
    data, _ = toy([3.0, 4.0], seed=1)
    b = np.array([1e3, 1e3])
    m = lm.marginalise(data, IVAR, G, b)
    flat = 0.5 * np.linalg.slogdet(2 * np.pi * m.covariance)[1]
    assert m.ln_penalty == pytest.approx(flat - np.sum(np.log(b)), abs=1e-10)


# %% box probability


@pytest.mark.parametrize("rho", [-0.99, -0.95, -0.7, 0.0, 0.5, 0.95, 0.99])
@pytest.mark.parametrize("mean", [(2.0, 3.0), (0.03, 0.02), (-1.0, 0.5), (-5.0, -4.0), (-8.0, -8.0), (-20.0, -20.0), (-12.0, 1.0), (7.0, 9.5), (9.5, 7.0), (3.0, -6.0)])
def test_box_2d_against_quadrature(rho, mean):
    """The deterministic 2-D box probability matches quadrature, also deep in the tails."""
    s = np.array([0.4, 0.3])
    cov = np.array([[s[0] ** 2, rho * s[0] * s[1]], [rho * s[0] * s[1], s[1] ** 2]])
    mean, upper = np.array(mean) * s, np.array([6.0, 7.5]) * s / 3
    ref = ref_box_2d(mean, cov, upper)
    assert lm.box_lnprob(mean, cov, upper) == pytest.approx(ref, abs=1e-4)


@pytest.mark.parametrize("mean", [(0.5, 0.3, 0.2), (0.02, -0.01, 0.03), (-0.6, 0.2, 0.4)])
def test_box_3d_against_quadrature(mean):
    """The QMC box probability of a 3-block meets |error ln P| < 0.01 against nested quadrature."""
    mean = np.array(mean)
    corr = np.array([[1.0, -0.6, 0.2], [-0.6, 1.0, -0.5], [0.2, -0.5, 1.0]])
    s = np.array([0.1, 0.15, 0.12])
    cov = corr * np.outer(s, s)
    upper = np.array([1.0, 1.2, 0.9])
    # condition the last two on the first and integrate the first with quadrature
    c12 = cov[1:, 0] / cov[0, 0]
    ccov = cov[1:, 1:] - np.outer(cov[1:, 0], cov[0, 1:]) / cov[0, 0]

    def f(x):
        return np.exp(stats.norm.logpdf(x, mean[0], s[0]) + lm.box_lnprob(mean[1:] + c12 * (x - mean[0]), ccov, upper[1:]))

    val, _ = integrate.quad(f, 0.0, upper[0], epsabs=0, epsrel=1e-10, limit=200)
    assert lm.box_lnprob(mean, cov, upper) == pytest.approx(np.log(val), abs=0.01)


def test_box_is_deterministic():
    """The same inputs give the same ln P, also for blocks of 3 or more."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + 4 * np.eye(4)
    mean, upper = rng.normal(size=4), np.full(4, 3.0)
    assert lm.box_lnprob(mean, cov, upper) == lm.box_lnprob(mean, cov, upper)


def test_blocks_from_shared_pixels():
    """Columns are grouped by shared fitted pixels, and the product over blocks is the full P_B."""
    x = np.linspace(0.0, 100.0, 1000)
    centres = [10.0, 40.0, 42.0, 70.0, 72.0, 74.0]
    templates = np.stack([np.exp(-0.5 * (x - c) ** 2) * (np.abs(x - c) < 5) for c in centres], axis=1)
    ivar = np.full(x.size, 1 / 0.05**2)
    truth = np.array([1.0, 0.8, 0.5, 0.6, 0.2, 0.9])
    data = templates @ truth + np.random.default_rng(4).normal(0, 0.05, x.size)
    upper = np.full(len(centres), 3.0)
    m = lm.marginalise(data, ivar, templates, upper)
    assert [list(b) for b in m.blocks] == [[0], [1, 2], [3, 4, 5]]
    full = stats.multivariate_normal.cdf(
        upper, m.amplitude, m.covariance, lower_limit=np.zeros(len(centres)), maxpts=10**6, abseps=1e-8,
        rng=np.random.default_rng(0),
    )
    lnpb = sum(lm.box_lnprob(m.amplitude[b], m.covariance[np.ix_(b, b)], upper[b]) for b in m.blocks)
    assert lnpb == pytest.approx(np.log(full), abs=0.01)


# %% edge cases


def test_column_without_data_drops_out():
    """A column with no fitted pixels leaves the penalty unchanged and gets zero amplitude."""
    data, _ = toy([0.3, 4.0], seed=1)
    extra = np.zeros((X.size, 1))
    extra[:10, 0] = 1.0
    ivar = IVAR.copy()
    ivar[:10] = 0.0
    both = lm.marginalise(data, ivar, np.hstack([G, extra]), np.append(W * C, 5.0))
    alone = lm.marginalise(data, ivar, G, W * C)
    assert both.ln_penalty == alone.ln_penalty
    assert both.amplitude[2] == 0.0
    assert [list(b) for b in both.blocks] == [[0, 1]]


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_nonpositive_bound_raises(bad):
    """A bound b = W c that is not positive (model continuum <= 0) raises ValueError."""
    data, _ = toy([0.3, 4.0], seed=1)
    with pytest.raises(ValueError):
        lm.marginalise(data, IVAR, G, np.array([6.0, bad]))


# %% truncated posterior mean


def test_truncated_mean_1d():
    """The 1-D mean matches scipy's truncated normal, also in the tail."""
    for mu, s, b in [(0.3, 0.1, 6.0), (-0.5, 0.1, 6.0), (5.98, 0.1, 6.0), (-3.0, 0.2, 1.0)]:
        ref = stats.truncnorm.mean(-mu / s, (b - mu) / s, loc=mu, scale=s)
        got = lm.truncated_mean(np.array([mu]), np.array([[s**2]]), np.array([b]), [np.array([0])])
        assert got[0] == pytest.approx(ref, rel=1e-8)


@pytest.mark.parametrize("name", CASES)
def test_truncated_mean_2d(name):
    """The 2-D mean matches the ratio of box integrals for each section 8 case."""
    data, lnlike = toy(CASES[name][0], seed=1)
    b = W * C
    m = lm.marginalise(data, IVAR, G, b)

    def integral(weight):
        return integrate.dblquad(
            lambda a2, a1: weight(a1, a2) * np.exp(lnlike(np.array([a1, a2])) - lnlike(m.amplitude)),
            0.0, b[0], 0.0, b[1], epsabs=1e-13, epsrel=1e-11,
        )[0]  # fmt: skip

    den = integral(lambda a1, a2: 1.0)
    ref = np.array([integral(lambda a1, a2: a1), integral(lambda a1, a2: a2)]) / den
    got = lm.truncated_mean(m.amplitude, m.covariance, b, m.blocks)
    assert np.allclose(got, ref, atol=1e-6)


def test_truncated_mean_3d():
    """The 3-D mean matches a large rejection sample within its Monte Carlo error."""
    mean = np.array([0.02, -0.01, 0.5])
    corr = np.array([[1.0, -0.6, 0.2], [-0.6, 1.0, -0.5], [0.2, -0.5, 1.0]])
    cov = corr * np.outer([0.1, 0.15, 0.12], [0.1, 0.15, 0.12])
    upper = np.array([1.0, 1.2, 0.55])
    draws = np.random.default_rng(5).multivariate_normal(mean, cov, size=4_000_000)
    keep = draws[np.all((draws >= 0) & (draws <= upper), axis=1)]
    ref, err = keep.mean(axis=0), keep.std(axis=0) / np.sqrt(len(keep))
    got = lm.truncated_mean(mean, cov, upper, [np.arange(3)])
    assert np.all(np.abs(got - ref) < 4 * err + 1e-3 * np.sqrt(np.diag(cov)))
