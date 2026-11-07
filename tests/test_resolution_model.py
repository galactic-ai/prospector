"""SpecModel predictions through a ResolutionMatrixSpectrum, with a fake SPS on a C3K_ER-like grid."""

import numpy as np
import pytest

from prospect.models import priors
from prospect.models.sedmodel import SpecModel
from prospect.observation import ResolutionMatrixSpectrum, Spectrum

CKMS = 2.998e5
LIB = 12.7  # km/s, library resolution of the fake SPS (as C3K_ER)
SIGMA, ELINE_SIGMA = 40.0, 30.0  # km/s
DPIX, NDIAG, WIDTH = 0.8, 11, 1.0  # data step (A), diagonals, row width (pixels)
LINES = (np.array([4862.7, 5008.2]), np.array([3e-7, 8e-7]))
rng = np.random.default_rng(0)
CENTRES, DEPTHS = rng.uniform(4800.0, 5200.0, 30), rng.uniform(0.05, 0.5, 30)


class FakeSPS:
    """Absorption-line continuum on a 10 km/s log grid with a constant library resolution."""

    def __init__(self, continuum=True, library_resolution=LIB):
        self.wave = np.exp(np.arange(np.log(3000.0), np.log(9000.0), 10.0 / CKMS))
        self.spectral_resolution = np.broadcast_to(library_resolution, self.wave.shape).copy()
        v = CKMS * np.log(self.wave[:, None] / CENTRES[None, :])
        depth = np.sum(DEPTHS * np.exp(-0.5 * (v / LIB) ** 2), axis=1)
        self.flux = 1e-8 * (1.0 - depth) if continuum else np.zeros(self.wave.size)

    def get_galaxy_spectrum(self, **params):
        return self.wave, self.flux.copy(), 1.0

    def get_galaxy_elines(self):
        return LINES[0], LINES[1].copy()


def make_model(lines=True, sigma=SIGMA, marginalize=False):
    """Return a SpecModel with fixed parameters and the two lines of LINES."""
    pars = {
        "zred": dict(N=1, isfree=False, init=0.0),
        "mass": dict(N=1, isfree=False, init=1.0),
        "lumdist": dict(N=1, isfree=False, init=1e-5),
        "sigma_smooth": dict(N=1, isfree=True, init=sigma, prior=priors.TopHat(mini=1.0, maxi=400.0)),
        "eline_sigma": dict(N=1, isfree=False, init=ELINE_SIGMA),
        "add_neb_emission": dict(N=1, isfree=False, init=lines),
        "nebemlineinspec": dict(N=1, isfree=False, init=False),
        "marginalize_elines": dict(N=1, isfree=False, init=marginalize),
    }
    model = SpecModel(pars)
    model.emline_info = np.array([(LINES[0][0], "Ba-beta 4861"), (LINES[0][1], "[O III] 5007")],
                                 dtype=[("wave", "f8"), ("name", "<U20")])
    model._use_eline = np.ones(2, dtype=bool)
    model.parse_elines()
    return model


@pytest.fixture(scope="module")
def observations():
    """A Gaussian-row ResolutionMatrixSpectrum and the Spectrum with the same Gaussian LSF."""
    n = 750
    wave = 4700.0 + DPIX * np.arange(n)
    d = np.arange(NDIAG) - NDIAG // 2
    k = np.exp(-0.5 * (d / WIDTH) ** 2)
    data = np.tile((k / k.sum())[::-1, None], (1, n))
    kw = dict(flux=np.full(n, 1e-8), uncertainty=np.full(n, 1e-10))
    rmat = ResolutionMatrixSpectrum(wavelength=wave, resolution_matrix=data, **kw)
    gauss = Spectrum(wavelength=wave, resolution=CKMS * DPIX * WIDTH / wave, **kw)
    rmat.rectify()
    gauss.rectify()
    return wave, rmat, gauss


def predict(model, obs, sps):
    return model.predict(model.theta.copy(), observations=[obs], sps=sps)[0][0]


def analytic_continuum(wave):
    """The fake continuum smoothed by the LOSVD and the Gaussian LSF, normalised to one."""
    out = np.ones_like(wave)
    for c, d in zip(CENTRES, DEPTHS):
        total = np.hypot(SIGMA, CKMS * DPIX * WIDTH / c)  # library removed, then added back
        out -= d * (LIB / total) * np.exp(-0.5 * ((wave / c - 1) * CKMS / total) ** 2)
    return out


def test_continuum_matches_analytic(observations):
    """The R path continuum is within 1e-3 of the analytic result (3.4e-4 measured 2026-11-06;
    the Gaussian-LSF path, which interpolates before its FFTs, gave 1.4e-3)."""
    wave, rmat, _ = observations
    out = predict(make_model(lines=False), rmat, FakeSPS())
    core = (wave > 4800.0) & (wave < 5200.0)
    truth = analytic_continuum(wave)
    scale = np.median(out[core] / truth[core])
    assert np.max(np.abs(out / scale - truth)[core]) < 1e-3


def test_line_matches_gaussian_path(observations):
    """A 30 km/s line has the Gaussian-LSF path's flux to 2e-3 and its analytic shape to 1e-2.

    Measured 2026-11-06: flux ratio 1.0007 and shape 3.2e-3 of the peak; the shape residual is
    the spline between the row samples of R, which a 0.6 pixel line probes.
    """
    wave, rmat, gauss = observations
    model, sps = make_model(), FakeSPS(continuum=False)
    out, ref = predict(model, rmat, sps), predict(model, gauss, sps)
    mu = LINES[0][1]
    sel = np.abs(wave - mu) < 6.0
    shape = np.exp(-0.5 * ((wave / mu - 1) * CKMS / np.hypot(ELINE_SIGMA, CKMS * DPIX / mu)) ** 2)
    amp = np.sum(out[sel] * shape[sel]) / np.sum(shape[sel] ** 2)
    assert abs(out[sel].sum() / ref[sel].sum() - 1) < 2e-3
    assert np.max(np.abs(out - amp * shape)[sel]) < 1e-2 * amp


def test_caches_after_prediction(observations):
    """Both lines are added and the caches match prospect's (response is one here)."""
    wave, rmat, _ = observations
    model = make_model()
    out = predict(model, rmat, FakeSPS())
    assert np.all(model._valid_eline)
    assert model._fix_eline_pixelmask.shape == wave.shape and model._fix_eline_pixelmask.any()
    assert model._fit_eline_pixelmask.shape == wave.shape and not model._fit_eline_pixelmask.any()
    assert np.allclose(model._sed * model._speccal, out)
    assert np.allclose(model._resolved_continuum + model._resolved_lines, model._sed)


def test_refuses_sigma_below_library(observations):
    """A LOSVD below the library resolution cannot be modelled and raises."""
    _, rmat, _ = observations
    with pytest.raises(ValueError, match="library"):
        predict(make_model(sigma=10.0), rmat, FakeSPS())


def test_refuses_varying_library_resolution(observations):
    """The library resolution must be constant where the model reaches fitted pixels."""
    _, rmat, _ = observations
    sps = FakeSPS()
    sps.spectral_resolution = np.linspace(10.0, 30.0, sps.wave.size)
    with pytest.raises(ValueError, match="varies"):
        predict(make_model(sigma=50.0), rmat, sps)


def test_refuses_line_marginalisation(observations):
    """Fitted (marginalised) lines are not supported through R."""
    _, rmat, _ = observations
    with pytest.raises(NotImplementedError, match="marginalization"):
        predict(make_model(marginalize=True), rmat, FakeSPS())
