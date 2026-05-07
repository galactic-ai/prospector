"""
NebSSPBasis — FastStepBasis subclass that uses cue.Emulator for nebular
emission lines (Li+24) instead of FSPS+Cloudy defaults.

Ported from bd-j/prospector@add_cue (commit 5f24db5) onto v2 main.

v1→v2 changes:
- import path: prospect.sources.ssp_basis → prospect.sources.galaxy_basis
- _line_specific_luminosity attribute already hooked in v2's
  galaxy_basis.SSPBasis.get_galaxy_elines (line ~144), so no override needed.

Limitations:
- use_stellar_ionizing=True path requires fit_log_linear_ionparam (undefined in
  add_cue branch); raises NotImplementedError. Use False for first pass.
"""

import numpy as np

from .galaxy_basis import FastStepBasis
from .fake_fsps import add_dust, add_igm

try:
    import cue
except ImportError:
    cue = None


__all__ = ["NebSSPBasis"]


class NebSSPBasis(FastStepBasis):
    """FastStepBasis subclass with Cue (Li+24) nebular emission emulator.

    Bypasses FSPS+Cloudy emission lines + nebular continuum + dust + IGM
    in favor of: Cue lines/continuum, prospect-level dust+IGM (fake_fsps).

    :param cue_kwargs: dict
        Forwarded to ``cue.Emulator(...)`` at construction.

    :param reserved_params: list
        Extends parent's reserved_params with ``dust1, dust2, dust3,
        add_dust_emission, add_igm_absorption, igm_factor, add_neb_emission,
        add_neb_continuum, nebemlineinspec, fagn, agn_tau``. These are
        handled here, not by FSPS.
    """

    def __init__(self, cue_kwargs=None, **kwargs):
        if cue is None:
            raise ImportError("cue not installed; pip install astro-cue")
        cue_kwargs = cue_kwargs or {}
        self.emul = cue.Emulator(**cue_kwargs)

        rp = ["dust1", "dust2", "dust3", "add_dust_emission",
              "add_igm_absorption", "igm_factor",
              "add_neb_emission", "add_neb_continuum", "nebemlineinspec",
              "fagn", "agn_tau"]
        reserved_params = list(kwargs.pop("reserved_params", [])) + rp
        super().__init__(reserved_params=reserved_params, **kwargs)

        for k in ["add_igm_absorption", "add_dust_emission",
                  "add_neb_emission", "nebemlineinspec"]:
            self.ssp.params[k] = False

    def get_galaxy_spectrum(self, **params):
        """Build tabular SFH, get FSPS continuum (no neb/dust/igm), layer Cue."""
        self.update(**params)
        if np.min(np.diff(10 ** self.params['agebins'])) < 1e6:
            raise ValueError("agebins spacing < 1 Myr would crash FSPS")

        mtot = self.params['mass'].sum()
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3
        self.ssp.set_tabular_sfh(time, sfr)

        wave, spec, lines = _get_spectrum(self.ssp, self.params, self.emul, tage=tmax)
        self._line_specific_luminosity = lines
        return wave, spec / mtot, self.ssp.stellar_mass / mtot


def _get_spectrum(ssp, params, emul, tage=0):
    """Returns (wave, sspec, lines) with Cue nebular replacing FSPS Cloudy.

    sspec includes: stellar continuum (FSPS) + Cue nebular continuum + dust + IGM.
    lines: emission line specific luminosity per young/old population.
    """
    add_neb = params.get("add_neb_emission", True)
    use_stars = params.get("use_stellar_ionizing", False)
    ewave = ssp.emline_wavelengths
    wave, _ = ssp.get_spectrum(tage=tage, peraa=False)

    # split stellar continuum into young + old populations
    young, old = ssp._csp_young_old
    csps = [young, old]
    lines = []
    for spec in csps:
        if add_neb:
            if use_stars:
                # extract Q_ion from young stellar SED to drive Cue
                ion_params = _fit_log_linear_ionparam(wave, spec)
                params.update(**ion_params)
            line_prediction = emul.predict_lines(**params)
            lines.append(line_prediction)
            spec += emul.predict_cont(wave, **params)
        else:
            lines.append(np.zeros_like(ewave))

    sspec, lines = add_dust(wave, csps, ewave, lines, **params)
    sspec = add_igm(wave, sspec, **params)
    return wave, sspec, lines


def _fit_log_linear_ionparam(wave, spec):
    """Derive Cue ionization params from stellar SED (Q_ion etc).

    NOT implemented on add_cue branch (commit 5f24db5). Stub raises until
    needed. For first-pass Cue integration use ``use_stellar_ionizing=False``
    and pass log_qion / log_OH / gas_logu directly via model params.
    """
    raise NotImplementedError(
        "fit_log_linear_ionparam not ported. Set use_stellar_ionizing=False "
        "and provide Cue ionization params directly in model template."
    )