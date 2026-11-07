"""Resolution matrices for spectra, resampled onto a fine model grid.

A spectrograph that extracts spectra by forward modelling (for example DESI) gives each spectrum a
banded resolution matrix R on its uniform extraction grid. Row i of R is the line spread function
of pixel i. The rows sum to one and the columns do not, so the model is multiplied by R itself.

Rebinning a model with unresolved lines onto the extraction grid before applying R biases it.
Instead each row of R is interpolated with a cubic spline onto a uniform fine model grid and
multiplied by the fine step over the extraction step (Guy et al. 2023, arXiv:2209.14482,
appendix D). The result R' applies the resolution and rebins in one step. R' keeps the row sums
of R and equals R on a smooth model. The spline is tied to zero, with zero slope, one pixel beyond
the outermost diagonals.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix, dia_matrix

__all__ = ["FINE_STEP", "banded_offsets", "banded_to_csr", "uniform_step", "fine_grid",
           "resampled_matrix"]

FINE_STEP = 0.05  # Angstrom


def banded_offsets(ndiag):
    """Return the diagonal offsets of a banded matrix.

    Parameters
    ----------
    ndiag : int
        Number of stored diagonals. It must be odd.

    Returns
    -------
    np.ndarray
        Offsets from ndiag//2 down to -(ndiag//2).
    """
    if ndiag % 2 != 1:
        raise ValueError(f"a banded matrix has an odd number of diagonals, not {ndiag}")
    return ndiag // 2 - np.arange(ndiag)


def banded_to_csr(data):
    """Return a banded matrix as a square sparse matrix.

    Parameters
    ----------
    data : np.ndarray
        Banded matrix of shape (ndiag, n), with R[j - offset_k, j] = data[k, j].

    Returns
    -------
    scipy.sparse.csr_matrix
        The matrix R, of shape (n, n).
    """
    data = np.asarray(data, dtype=np.float64)
    ndiag, n = data.shape
    return dia_matrix((data, banded_offsets(ndiag)), shape=(n, n)).tocsr()


def uniform_step(wave, rtol=1e-6):
    """Return the step of a uniform wavelength grid.

    Parameters
    ----------
    wave : np.ndarray
        Increasing wavelengths in Angstrom.
    rtol : float
        Largest allowed relative deviation of any step from the mean step.

    Returns
    -------
    float
        The step in Angstrom.

    Raises
    ------
    ValueError
        If the grid is not uniform to ``rtol``. A float32 copy of a float64 grid can fail at
        the default tolerance.
    """
    wave = np.asarray(wave)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("a wavelength grid must be one dimensional with two or more points")
    step = (float(wave[-1]) - float(wave[0])) / (wave.size - 1)
    if step <= 0 or not np.allclose(np.diff(wave.astype(np.float64)), step, rtol=rtol, atol=0):
        raise ValueError(f"the wavelength grid is not uniform to a relative {rtol:g}")
    return step


def fine_grid(wave, ndiag, step=FINE_STEP):
    """Return a uniform fine grid that covers ``wave`` and the reach of the matrix rows.

    Parameters
    ----------
    wave : np.ndarray
        The uniform extraction grid of the data in Angstrom.
    ndiag : int
        Number of diagonals of the matrix.
    step : float
        Step of the fine grid in Angstrom.

    Returns
    -------
    np.ndarray
        The fine grid, from ndiag//2 + 1 pixels below ``wave[0]`` to as far above ``wave[-1]``.
    """
    dpix = uniform_step(wave)
    pad = (ndiag // 2 + 1) * dpix
    return np.arange(wave[0] - pad, wave[-1] + pad + step / 2, step)


def resampled_matrix(wave, data, fine):
    """Return R', the rows of a banded resolution matrix resampled onto a fine grid.

    Parameters
    ----------
    wave : np.ndarray
        The uniform extraction grid in Angstrom, in float64.
    data : np.ndarray
        Banded matrix of shape (ndiag, n), with R[j - offset_k, j] = data[k, j].
    fine : np.ndarray
        The uniform fine model grid, from ``fine_grid``.

    Returns
    -------
    scipy.sparse.csr_matrix
        R' of shape (n, fine.size). R' @ m is the extracted model for a model m on ``fine``.

    Raises
    ------
    ValueError
        If a grid is not uniform, the shapes do not match, or ``fine`` does not cover the rows.
    """
    data = np.asarray(data, dtype=np.float64)
    ndiag, n = data.shape
    offs = -banded_offsets(ndiag)
    c = ndiag // 2
    if wave.shape != (n,):
        raise ValueError(f"wave has shape {wave.shape}, the matrix has {n} columns")
    dpix = uniform_step(wave)
    step = uniform_step(fine)
    reach = (c + 1) * dpix
    if fine[0] > wave[0] - reach + step / 2 or fine[-1] < wave[-1] + reach - step / 2:
        raise ValueError("the fine grid does not cover the reach of the matrix rows")
    knots = np.arange(-c - 1, c + 2) * dpix
    rows, cols, vals = [], [], []
    for i in range(n):
        # row i holds R[i, j] at j = i + offs, stored at k = c - (j - i)
        j = i + offs
        ok = (j >= 0) & (j < n)
        y = np.zeros(ndiag)
        y[ok] = data[(c - offs)[ok], j[ok]]
        lam = wave[i] + knots
        cs = CubicSpline(lam, np.concatenate([[0.0], y, [0.0]]), bc_type=((1, 0.0), (1, 0.0)))
        f0, f1 = np.searchsorted(fine, [lam[0], lam[-1]])
        rows.append(np.full(f1 - f0, i))
        cols.append(np.arange(f0, f1))
        vals.append(cs(fine[f0:f1]) * step / dpix)
    return csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, fine.size)
    )

