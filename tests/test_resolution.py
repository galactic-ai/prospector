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
