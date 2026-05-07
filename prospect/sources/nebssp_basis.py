import numpy as np
from pkg_resources import resource_filename

import fsps
from .galaxy_basis import FastStepBasis
from .fake_fsps import add_dust, add_igm


try:
    from cue import Emulator
    from cue.utils import fit_4loglinear_ionparam        
    from cue.utils import line_lam
except ImportError:
    raise ImportError("cue is required to use NebStepBasis. Install with `pip install astro-cue`.")


__all__ = ["NebStepBasis"]

cue_keys = [
    "ionspec_index1", "ionspec_index2", "ionspec_index3", "ionspec_index4",
    "ionspec_logLratio1", "ionspec_logLratio2", "ionspec_logLratio3",
    "gas_logu", "gas_logn", "gas_logz", "gas_logno", "gas_logco",
]

def _predict_lines(theta, emul: Emulator):
    """1D line spec for one theta."""
    out = emul.predict_lines(theta=np.atleast_2d(theta))
    return np.squeeze(np.atleast_2d(out)[0])

def _predict_cont(theta, wave, emul: Emulator):
    """1D continuum for one theta."""
    out = emul.predict_cont(theta=np.atleast_2d(theta), wave=wave)
    return np.squeeze(np.atleast_2d(out)[0])

class NebStepBasis(FastStepBasis):
    """FastStepBasis with nebular emission lines and 
    continuum from the Cue (Li+24) emulator.
    
    Replaces CLOUDY+FSPS nebular emission with the Cue emulator predictions.
    Dust+IGM applied via `prospect.sources.fake_fsps`.
    """

    def __init__(self, cue_kwargs=None, **kwargs):
        if Emulator is None:
            raise ImportError("cue is required to use NebStepBasis. Install with `pip install astro-cue`.")
        
        rp = ["dust1", "dust2", "dust3", "add_dust_emission",
            "add_igm_absorption", "igm_factor",
            "add_neb_emission", "add_neb_continuum", "nebemlineinspec",
            "fagn", "agn_tau"]
        reserved_params = list(kwargs.pop("reserved_params", [])) + rp
        super().__init__(reserved_params=reserved_params, **kwargs)
        for k in ["add_igm_absorption", "add_dust_emission",
                "add_neb_emission", "nebemlineinspec"]:
            self.ssp.params[k] = False

        cue_kwargs = cue_kwargs or {}
        self.emul = Emulator(**cue_kwargs)

        # warm up TF graph + load weights at __init__ to avoid first-call latency
        _theta_default = [19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2,
                        -2.5, 2.0, 0.0, 0.0, 0.0]
        _ = _predict_lines(_theta_default, self.emul)
        _ = _predict_cont(_theta_default, self.ssp.wavelengths, self.emul)

        # Cue's emission line wav array
        self.emline_wavelengths = np.asarray(line_lam)


    def get_galaxy_spectrum(self, **params):
        """Build Tabular SFH, get the FSPS spectrum, add Cue"""
        self.update(**params)
        if np.min(np.diff(10 ** self.params['agebins'])) < 1e6:
            raise ValueError("agebins spacing < 1 Myr would crash FSPS")
 
        mtot = self.params['mass'].sum()
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3
        self.ssp.set_tabular_sfh(time, sfr)
 
        wave, spec, lines = _get_spectrum(
            self.ssp, self.params, self.emul, self.emline_wavelengths, tage=tmax)
        self._line_specific_luminosity = lines
        return wave, spec / mtot, self.ssp.stellar_mass / mtot

    def get_galaxy_elines(self):
        """Override to return Cue line luminosities instead of FSPS."""
        ewave = self.emline_wavelengths
        elum = getattr(self, "_line_specific_luminosity", None)

        if elum is None:
            ewave = self.ssp.emline_wavelengths
            elum = self.ssp.emline_luminosity.copy()
        elum = np.asarray(elum)

        if elum.ndim > 1:
            elum = elum[0]
        if self.ssp.params["sfh"] == 3:
            mass = np.sum(self.params.get("mass", 1.0))
            elum = elum / mass
        return ewave, elum


def _get_spectrum(ssp, params, emul, ewave, tage=0):
    """Get FSPS spectrum, then add Cue lines and continuum. 
    And then add dust+IGM."""

    add_neb = params.get("add_neb_emission", False)
    use_stars = params.get("use_stellar_ionizing", False)
    wave, total_spec = ssp.get_spectrum(tage=tage, peraa=True)
    csps = [total_spec, np.zeros_like(total_spec)]
    lines_list = [np.zeros_like(ewave), np.zeros_like(ewave)]

    if add_neb:
        if use_stars:
            params.update(**fit_4loglinear_ionparam(wave, total_spec))
        
        cue_params = {k: params[k] for k in cue_keys}
        theta = np.array(list(cue_params.values()))
        line_pred = _predict_lines(theta, emul)
        lines_list = [line_pred, np.zeros_like(ewave)]
 
        mask912 = wave >= 912
        csps[0][mask912] += _predict_cont(theta, wave[mask912], emul)
 
    sspec, lines = add_dust(wave, csps, ewave, lines_list, **params)
    sspec = add_igm(wave, sspec, **params)
    return wave, sspec, lines