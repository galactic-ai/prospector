"""Banded resolution matrix R and its resampled form R' on synthetic inputs."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix, dia_matrix

from prospect.observation import resolution as rm

NDIAG = 11
DPIX = 0.8
CKMS = 299792.458


def synthetic_banded(n=300, ndiag=NDIAG, seed=0):
    """Return a grid and a banded matrix with skewed rows of varying width and row sum.

    Row i is a skewed Gaussian of width 0.55-0.95 pixels, scaled to a row sum of 0.95-1.05, so
    the rows are not symmetric, not normalised to one, and differ from their columns.
    """
    rng = np.random.default_rng(seed)
    wave = 5000.0 + DPIX * np.arange(n)
    c = ndiag // 2
    d = np.arange(-c, c + 1)
    width = 0.55 + 0.4 * rng.uniform(size=n)
    skew = 0.3 * rng.uniform(-1, 1, size=n)
    norm = 0.95 + 0.1 * rng.uniform(size=n)
    dense = np.zeros((n, n))
    for i in range(n):
        k = np.exp(-0.5 * (d / width[i]) ** 2) * (1 + skew[i] * np.tanh(d))
        j = i + d
        ok = (j >= 0) & (j < n)
        dense[i, j[ok]] = norm[i] * k[ok] / k.sum()
    data = np.zeros((ndiag, n))
    for kk, off in enumerate(rm.banded_offsets(ndiag)):
        j = np.arange(n)
        i = j - off
        ok = (i >= 0) & (i < n)
        data[kk, j[ok]] = dense[i[ok], j[ok]]
    return wave, data, dense


def gaussian_banded(n=300, ndiag=NDIAG, width=0.8):
    """Return a grid and a banded matrix whose rows are one normalised Gaussian (a convolution)."""
    wave = 5000.0 + DPIX * np.arange(n)
    d = np.arange(ndiag) - ndiag // 2
    k = np.exp(-0.5 * (d / width) ** 2)
    data = np.tile((k / k.sum())[::-1, None], (1, n))
    return wave, data


def narrow_line(fine, mu, sigma_kms):
    """Return a unit-flux Gaussian line on ``fine``, in flux per Angstrom."""
    s = mu * sigma_kms / CKMS
    g = np.exp(-0.5 * ((fine - mu) / s) ** 2)
    return g / (g.sum() * (fine[1] - fine[0]))


def test_banded_convention_matches_definition_and_scipy():
    """R[j - offset_k, j] = data[k, j] with offsets ndiag//2 down to -(ndiag//2), as scipy dia."""
    wave, data, dense = synthetic_banded()
    offsets = NDIAG // 2 - np.arange(NDIAG)
    assert np.array_equal(rm.banded_offsets(NDIAG), offsets)
    scipy_dense = dia_matrix((data, offsets), shape=dense.shape).toarray()
    assert np.array_equal(scipy_dense, dense)
    assert np.array_equal(rm.banded_to_csr(data).toarray(), dense)
    # data[:, j] holds column j, from row j + ndiag//2 at k = 0 down to row j - ndiag//2
    j = 100
    assert np.array_equal(data[:, j], dense[j - offsets, j])
    with pytest.raises(ValueError, match="odd"):
        rm.banded_offsets(10)


def test_rprime_keeps_row_sums():
    """The rows of R' sum to the rows of R, which are not normalised to one here."""
    wave, data, dense = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    rp = rm.resampled_matrix(wave, data, fine)
    assert rp.shape == (wave.size, fine.size)
    assert np.allclose(np.asarray(rp.sum(axis=1)).ravel(), dense.sum(axis=1), rtol=0, atol=1e-5)


def test_rprime_equals_r_on_smooth_input():
    """On a smooth model R' gives what R gives on the model sampled at the pixels."""
    wave, data, dense = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    rp = rm.resampled_matrix(wave, data, fine)

    def model(w):
        return 1.0 + 0.3 * np.sin(2 * np.pi * (w - 5000.0) / 150.0) + 1e-4 * (w - 5000.0)

    out, ref = rp @ model(fine), dense @ model(wave)
    assert np.max(np.abs(out - ref)) < 1e-5 * np.max(np.abs(ref))


def test_unresolved_line_on_a_pixel_gives_its_column():
    """A delta function at pixel j, on the fine grid, maps to column j of R times step / 0.8 A."""
    wave, data, dense = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    step = fine[1] - fine[0]
    rp = rm.resampled_matrix(wave, data, fine)
    for j in (0, 3, 150, wave.size - 1):
        f = int(np.argmin(np.abs(fine - wave[j])))
        assert abs(fine[f] - wave[j]) < 1e-9
        m = np.zeros(fine.size)
        m[f] = 1.0
        # the fine points sit within 1e-9 A of the pixels, which the spline slope turns into 1e-9
        assert np.allclose(rp @ m * DPIX / step, dense[:, j], rtol=0, atol=1e-8)


def test_rprime_requires_uniform_grid():
    """A float32, uneven or too short grid is refused."""
    wave, data, _ = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    with pytest.raises(ValueError, match="not uniform"):
        rm.resampled_matrix(wave.astype(np.float32), data, fine)
    uneven = wave.copy()
    uneven[50] += 0.1 * DPIX
    with pytest.raises(ValueError, match="not uniform"):
        rm.resampled_matrix(uneven, data, fine)
    with pytest.raises(ValueError, match="cover"):
        rm.resampled_matrix(wave, data, fine[20:])
    w32 = np.linspace(3600.0, 9824.0, 7781).astype(np.float32)
    with pytest.raises(ValueError, match="not uniform"):
        rm.uniform_step(w32)


def test_offpixel_line_keeps_flux_and_centroid():
    """A 10 km/s line between pixels keeps its flux and centroid under R' (Guy et al. 2023 app. D).

    With one normalised row kernel R is a flux-conserving convolution. Rebinning the line onto the
    pixels before R distorts its observed profile by 6-10% of the peak and moves its centroid by up
    to 0.08 A (measured 2026-11-06); R' keeps the centroid to 1e-4 A.
    """
    wave, data = gaussian_banded()
    fine = rm.fine_grid(wave, NDIAG)
    step = fine[1] - fine[0]
    rp = rm.resampled_matrix(wave, data, fine)
    r = rm.banded_to_csr(data)
    edges = np.r_[wave - DPIX / 2, wave[-1] + DPIX / 2]
    for frac in (0.1, 0.25, 0.45):
        mu = wave[150] + frac * DPIX
        m = narrow_line(fine, mu, 10.0)
        out = rp @ m
        assert abs(out.sum() * DPIX - 1.0) < 1e-4
        assert abs((wave * out).sum() / out.sum() - mu) < 1e-4
        binned = np.histogram(fine, bins=edges, weights=m * step)[0] / DPIX
        rebinned = r @ binned
        assert np.max(np.abs(out - rebinned)) > 0.05 * out.max()


def test_fine_step_is_converged():
    """At 0.05 A a 10 km/s line through R' is within 1e-5 of its peak of the 0.005 A result.

    Measured 2026-11-06, worst over 20 line positions: 4.4e-7 at 0.05 A and 7.5e-6 at 0.1 A near
    5120 A; 6.0e-7 and 1.1e-5 near 3720 A, where a 10 km/s line is narrowest in Angstrom.
    """
    wave, data, _ = synthetic_banded()
    ref_fine = rm.fine_grid(wave, NDIAG, step=0.005)
    ref_rp = rm.resampled_matrix(wave, data, ref_fine)
    fine = rm.fine_grid(wave, NDIAG, step=rm.FINE_STEP)
    rp = rm.resampled_matrix(wave, data, fine)
    for frac in np.linspace(0.0, 0.95, 20):
        mu = wave[150] + frac * DPIX
        ref = ref_rp @ narrow_line(ref_fine, mu, 10.0)
        out = rp @ narrow_line(fine, mu, 10.0)
        assert np.max(np.abs(out - ref)) < 1e-5 * ref.max()


def test_zero_tied_spline_keeps_row_sums_and_free_ends_do_not():
    """Tying each row's spline to zero one pixel beyond the band keeps the row sums of R.

    A not-a-knot spline through the band alone overshoots past its ends and changes the row sums
    by ~3e-3 (measured 2026-11-06), which is why the ends are tied.
    """
    wave, data, dense = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    tied = rm.resampled_matrix(wave, data, fine)

    ndiag, n = data.shape
    offs, c = -rm.banded_offsets(ndiag), ndiag // 2
    step = rm.uniform_step(fine)
    rows, cols, vals = [], [], []
    for i in range(n):
        j = i + offs
        ok = (j >= 0) & (j < n)
        y = np.zeros(ndiag)
        y[ok] = data[(c - offs)[ok], j[ok]]
        lam = wave[i] + np.arange(-c, c + 1) * DPIX
        f0, f1 = np.searchsorted(fine, [lam[0], lam[-1]])
        rows.append(np.full(f1 - f0, i))
        cols.append(np.arange(f0, f1))
        vals.append(CubicSpline(lam, y)(fine[f0:f1]) * step / DPIX)
    free = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, fine.size))

    rowsum = dense.sum(axis=1)
    assert np.max(np.abs(np.asarray(tied.sum(axis=1)).ravel() - rowsum)) < 1e-10
    assert np.max(np.abs(np.asarray(free.sum(axis=1)).ravel() - rowsum)) > 1e-4


def resolution_spectrum(data, mask=None):
    """Return a rectified ResolutionMatrixSpectrum of unit flux on the 5000 A test grid."""
    from prospect.observation import ResolutionMatrixSpectrum

    n = data.shape[1]
    kw = dict(mask=mask) if mask is not None else {}
    obs = ResolutionMatrixSpectrum(wavelength=5000.0 + DPIX * np.arange(n), resolution_matrix=data,
                                   flux=np.ones(n), uncertainty=0.1 * np.ones(n), **kw)
    obs.rectify()
    return obs


def test_spectrum_matches_gaussian_lsf():
    """With one Gaussian row kernel, R' smoothing equals Spectrum's Gaussian LSF of the same width.

    The residual, 2.4e-4 on a sine of amplitude 0.3 (2026-11-06), comes from sampling the kernel
    at the pixels.
    """
    from prospect.observation import Spectrum

    wave, data = gaussian_banded(n=400, width=1.0)
    obs = resolution_spectrum(data)
    gauss = Spectrum(wavelength=wave, resolution=DPIX / wave * CKMS, flux=np.ones(wave.size),
                     uncertainty=0.1 * np.ones(wave.size))
    gauss.rectify()
    mw = np.arange(4950.0, 5400.0, 0.1)
    mf = 1 + 0.3 * np.sin(2 * np.pi * (mw - 5000.0) / 40.0)
    out = obs.instrumental_smoothing(mw, mf)
    ref = gauss.instrumental_smoothing(mw, mf, libres=np.zeros_like(mw))
    assert np.max(np.abs(out - ref)[20:-20]) < 1e-3


def test_spectrum_caches_and_pickles_without_rprime():
    """R' is built once, left out of the pickle, and rebuilt to the same result."""
    import pickle

    _, data = gaussian_banded(n=200)
    obs = resolution_spectrum(data)
    first = obs.resampled()
    assert obs.resampled() is first
    copy = pickle.loads(pickle.dumps(obs))
    assert copy._rprime is None
    mw = np.arange(4950.0, 5200.0, 0.1)
    mf = 1 + 1e-3 * (mw - 5000.0)
    assert np.array_equal(copy.instrumental_smoothing(mw, mf), obs.instrumental_smoothing(mw, mf))


def test_spectrum_check_rows_uses_the_mask():
    """A zero row of R raises on a fitted pixel and is ignored on a masked one."""
    _, data = gaussian_banded(n=200)
    for kk, off in enumerate(rm.banded_offsets(NDIAG)):
        data[kk, 50 + off] = 0.0
    resolution_spectrum(data, mask=np.arange(200) != 50).check_rows()
    with pytest.raises(ValueError, match="row sums"):
        resolution_spectrum(data).check_rows()


def test_spectrum_refuses_bad_inputs():
    """Gaussian resolution, a missing or float32 grid, a wrong matrix shape and libres != 0."""
    from prospect.observation import ResolutionMatrixSpectrum

    wave, data = gaussian_banded(n=200)
    kw = dict(flux=np.ones(200), uncertainty=np.ones(200))
    with pytest.raises(ValueError, match="resolution_matrix, not resolution"):
        ResolutionMatrixSpectrum(wavelength=wave, resolution_matrix=data, resolution=60.0, **kw)
    with pytest.raises(ValueError, match="needs wavelength"):
        ResolutionMatrixSpectrum(resolution_matrix=data, **kw)
    with pytest.raises(ValueError, match="not uniform"):
        ResolutionMatrixSpectrum(wavelength=wave.astype(np.float32), resolution_matrix=data, **kw)
    with pytest.raises(ValueError, match="expected"):
        ResolutionMatrixSpectrum(wavelength=wave, resolution_matrix=data[:, :-1], **kw)
    obs = resolution_spectrum(data)
    mw = np.arange(4950.0, 5200.0, 0.1)
    with pytest.raises(ValueError, match="libres"):
        obs.instrumental_smoothing(mw, np.ones_like(mw), libres=np.full_like(mw, 10.0))
    with pytest.raises(ValueError, match="cover"):
        obs.instrumental_smoothing(mw[600:], np.ones(mw.size - 600))  # starts at 5010 A


def raw_profile(w, mu, sigma):
    """Return SpecModel's unnormalised Gaussian line profile per Hz at ``w``."""
    dv = rm.CKMS * (w / mu - 1)
    g = np.exp(-0.5 * (dv / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return g * rm.CKMS * w**2 / (rm.LIGHTSPEED * mu)


@pytest.mark.parametrize("sigma", [10.0, 45.0, 800.0])
def test_line_profile_has_unit_flux_in_frequency(sigma):
    """The profile sums to one over frequency, is centred on mu, and has width mu sigma / c."""
    fine = np.arange(4000.0, 7000.0, rm.FINE_STEP)
    mu = 5008.24
    sl, g = rm.line_profile(fine, mu, sigma)
    w = fine[sl]
    dnu = rm.LIGHTSPEED / w**2 * (fine[-1] - fine[0]) / (fine.size - 1)
    assert np.isclose(np.sum(g * dnu), 1.0, rtol=0, atol=1e-12)
    assert np.allclose(g, raw_profile(w, mu, sigma), rtol=1e-9, atol=0)
    assert abs(np.sum(w * g * dnu) - mu) < 1e-6 * mu * sigma / rm.CKMS + 1e-9
    var = np.sum((w - mu) ** 2 * g * dnu)
    assert np.isclose(np.sqrt(var), mu * sigma / rm.CKMS, rtol=1e-6)


def test_line_profile_at_grid_edge_and_too_narrow():
    """A line cut by the grid keeps its inside values; a line narrower than the step is refused."""
    fine = np.arange(4000.0, 7000.0, rm.FINE_STEP)
    cut = fine[: np.searchsorted(fine, 5010.0)]
    mu = 5008.24
    sl_full, g_full = rm.line_profile(fine, mu, 100.0)
    sl_cut, g_cut = rm.line_profile(cut, mu, 100.0)
    assert sl_cut.start == sl_full.start and sl_cut.stop == cut.size
    assert np.allclose(g_cut, g_full[: sl_cut.stop - sl_cut.start], rtol=1e-12, atol=0)
    assert rm.line_profile(fine, 3000.0, 100.0)[1].size == 0
    with pytest.raises(ValueError, match="narrower"):
        rm.line_profile(fine, mu, 1.0)
    assert rm.line_profile(fine, 1000.0, 1.0)[1].size == 0
