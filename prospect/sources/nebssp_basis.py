### SSP for cuejax
import numpy as np
from pkg_resources import resource_filename

import os
from scipy.interpolate import interp1d

import fsps
from .galaxy_basis import SSPBasis, FastStepBasis, CSPSpecBasis
from .fake_fsps import add_dust, add_igm, idx

try:
    from cuejax.emulator import Emulator, fast_line_prediction, fast_cont_prediction
    from cuejax.utils import fit_4loglinear_ionparam
except:
    pass


__all__ = ["NebSSPBasis", "NebStepBasis", "NebCSPBasis"]



class NebSSPBasis(SSPBasis):

    """This is a class that wraps the fsps.StellarPopulation object, which is
    used for producing SSPs.  The ``fsps.StellarPopulation`` object is accessed
    as ``SSPBasis().ssp``.

    This class allows for the custom calculation of relative SSP weights (by
    overriding ``all_ssp_weights``) to produce spectra from arbitrary composite
    SFHs. Alternatively, the entire ``get_galaxy_spectrum`` method can be
    overridden to produce a galaxy spectrum in some other way, for example
    taking advantage of weight calculations within FSPS for tabular SFHs or for
    parameteric SFHs.

    The base implementation here produces an SSP interpolated to the age given
    by ``tage``, with initial mass given by ``mass``.  However, this is much
    slower than letting FSPS calculate the weights, as implemented in
    :py:class:`FastSSPBasis`.

    Furthermore, smoothing, redshifting, and filter projections are handled
    outside of FSPS, allowing for fast and more flexible algorithms.

    :param reserved_params:
        These are parameters which have names like the FSPS parameters but will
        not be passed to the StellarPopulation object because we are overriding
        their functionality using (hopefully more efficient) custom algorithms.
    """

    def __init__(self, zcontinuous=1, reserved_params=['tage', 'sigma_smooth'],
                 interp_type='logarithmic', flux_interp='linear',
                 mint_log=-3, compute_vega_mags=False,
                 cue_kwargs={},
                 **kwargs):
        """
        :param interp_type: (default: "logarithmic")
            Specify whether to linearly interpolate the SSPs in log(t) or t.
            For the latter, set this to "linear".

        :param flux_interp': (default: "linear")
            Whether to compute the final spectrum as \sum_i w_i f_i or
            e^{\sum_i w_i ln(f_i)}.  Basically you should always do the former,
            which is the default.

        :param mint_log: (default: -3)
            The log of the age (in years) of the youngest SSP.  Note that the
            SSP at this age is assumed to have the same spectrum as the minimum
            age SSP avalibale from fsps.  Typically anything less than 4 or so
            is fine for this parameter, since the integral converges as log(t)
            -> -inf

        :param reserved_params:
            These are parameters which have names like the FSPS parameters but
            will not be passed to the StellarPopulation object because we are
            overriding their functionality using (hopefully more efficient)
            custom algorithms.
        """
        self.interp_type = interp_type
        self.mint_log = mint_log
        self.flux_interp = flux_interp
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        # we do these now
        rp = ["dust1", "dust2", "dust3", "add_dust_emission",
              "add_igm_absorption", "igm_factor",
              "add_neb_emission", "add_neb_continuum", "nebemlineinspec",
              "fagn", "agn_tau"]
        reserved_params = kwargs.pop("reserved_params", []) + rp
        super().__init__(reserved_params=reserved_params, **kwargs)
        for k in ["add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"]:
            self.ssp.params[k] = False            
        self.ssp.params['sfh'] = 0
        self.reserved_params = reserved_params
        self.params = {}
        self.update(**kwargs)
        
        self.emul = Emulator(**cue_kwargs) # gas_logqion is fixed to 49.1
        # compile the functions first to speed up prediction, using an initial set of cue parameters in order
        _ = fast_line_prediction([19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, -2.5, 2.0, 0.0, 0.0, 0.0], self.emul)
        _ = fast_cont_prediction([19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, -2.5, 2.0, 0.0, 0.0, 0.0], 
                                 self.ssp.wavelengths,
                                 self.emul, unit='Lsun/Hz')
        self.emline_wavelengths = np.genfromtxt(resource_filename("cuejax", "data/cue_emlines_info.dat"),
                                                dtype=[('wave', 'f8'), ('name', '<U20')],
                                                delimiter=',')['wave']

    def get_galaxy_spectrum(self, **params):
        """Update parameters, then get the SSP spectrum

        Returns
        -------
        wave : ndarray
            Restframe avelength in angstroms.

        spectrum : ndarray
            Spectrum in units of Lsun/Hz per solar mass formed.

        mass_fraction : float
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)
        wave, spec, lines = get_spectrum(self.ssp, self.params, self.emul, self.emline_wavelengths, tage=float(self.params['tage']))
        self._line_specific_luminosity = lines
        
        # mimic the "nebemlineinspec" function in FSPS
        if self.params.get("nebemlineinspec", False):
            if self.ssp.params["smooth_velocity"] == True:
                dlam = self.emline_wavelengths*self.ssp.params["sigma_smooth"]/2.9979E18*1E13 #smoothing variable is in km/s
            else:
                dlam = self.ssp.params["sigma_smooth"] #smoothing variable is in AA
            nearest_id = np.searchsorted(wave, self.emline_wavelengths)
            neb_res_min = wave[nearest_id]-wave[nearest_id-1]
            dlam = np.max([dlam,neb_res_min*2], axis=0)
            gaussnebarr = [1./np.sqrt(2*np.pi)/dlam[i]*np.exp(-(wave-self.emline_wavelengths[i])**2/2/dlam[i]**2) \
            /2.9979E18*self.emline_wavelengths[i]**2 for i in range(len(lines))]
            for i in range(len(lines)):
                spec += lines[i]*gaussnebarr[i]

        return wave, spec, self.ssp.stellar_mass

    def get_galaxy_elines(self):
        """Get the wavelengths and specific emission line luminosity of the nebular emission lines
        predicted by FSPS. These lines are in units of Lsun/solar mass formed.
        This assumes that `get_galaxy_spectrum` has already been called.

        :returns ewave:
            The *restframe* wavelengths of the emission lines, AA

        :returns elum:
            Specific luminosities of the nebular emission lines,
            Lsun/stellar mass formed
        """
        ewave = self.emline_wavelengths
        # This allows subclasses to set their own specific emission line
        # luminosities within other methods, e.g., get_galaxy_spectrum, by
        # populating the `_specific_line_luminosity` attribute.
        elum = getattr(self, "_line_specific_luminosity", None).copy()
        
        if elum is None:
            ewave = self.ssp.emline_wavelengths
            elum = self.ssp.emline_luminosity.copy()
        if elum.ndim > 1:
            elum = elum[0]
        if self.ssp.params["sfh"] == 3:
            # tabular sfh
            mass = np.sum(self.params.get('mass', 1.0))
            elum /= mass

        return ewave, elum

    
class NebStepBasis(FastStepBasis):

    """This is a class that wraps the fsps.StellarPopulation object, which is
    used for producing SSPs.  The ``fsps.StellarPopulation`` object is accessed
    as ``SSPBasis().ssp``.

    This class allows for the custom calculation of relative SSP weights (by
    overriding ``all_ssp_weights``) to produce spectra from arbitrary composite
    SFHs. Alternatively, the entire ``get_galaxy_spectrum`` method can be
    overridden to produce a galaxy spectrum in some other way, for example
    taking advantage of weight calculations within FSPS for tabular SFHs or for
    parameteric SFHs.

    The base implementation here produces an SSP interpolated to the age given
    by ``tage``, with initial mass given by ``mass``.  However, this is much
    slower than letting FSPS calculate the weights, as implemented in
    :py:class:`FastSSPBasis`.

    Furthermore, smoothing, redshifting, and filter projections are handled
    outside of FSPS, allowing for fast and more flexible algorithms.

    :param reserved_params:
        These are parameters which have names like the FSPS parameters but will
        not be passed to the StellarPopulation object because we are overriding
        their functionality using (hopefully more efficient) custom algorithms.
    """
    def __init__(self, zcontinuous=1, reserved_params=['tage', 'sigma_smooth'],
                 interp_type='logarithmic', flux_interp='linear',
                 mint_log=-3, compute_vega_mags=False,
                 cue_kwargs={},
                 **kwargs):
        
        self.interp_type = interp_type
        self.mint_log = mint_log
        self.flux_interp = flux_interp
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        # we do these now
        rp = ["dust1", "dust2", "dust3", "add_dust_emission",
              "add_igm_absorption", "igm_factor",
              "add_neb_emission", "add_neb_continuum", "nebemlineinspec",
              "fagn", "agn_tau"]
        reserved_params = kwargs.pop("reserved_params", []) + rp
        super().__init__(reserved_params=reserved_params, **kwargs)
        for k in ["add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"]:
            self.ssp.params[k] = False            
        self.ssp.params['sfh'] = 0
        self.reserved_params = reserved_params
        self.params = {}
        self.update(**kwargs)

        self.emul = Emulator(**cue_kwargs) # gas_logqion is fixed to 49.1
        # compile the functions first to speed up prediction, using an initial set of cue parameters in order
        _ = fast_line_prediction([19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, -2.5, 2.0, 0.0, 0.0, 0.0], self.emul)
        _ = fast_cont_prediction([19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, -2.5, 2.0, 0.0, 0.0, 0.0], 
                                 self.ssp.wavelengths,
                                 self.emul, unit='Lsun/Hz')
        self.emline_wavelengths = np.genfromtxt(resource_filename("cuejax", "data/cue_emlines_info.dat"),
                                                dtype=[('wave', 'f8'), ('name', '<U20')],
                                                delimiter=',')['wave']

    def get_galaxy_spectrum(self, **params):
        """Construct the tabular SFH and feed it to the ``ssp``.
        """
        self.update(**params)
        # --- check to make sure agebins have minimum spacing of 1million yrs ---
        #       (this can happen in flex models and will crash FSPS)
        if np.min(np.diff(10**self.params['agebins'])) < 1e6:
            raise ValueError

        mtot = self.params['mass'].sum()
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3  # Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(time, sfr)

        wave, spec, lines = get_spectrum(self.ssp, self.params, self.emul, self.emline_wavelengths, tage=tmax)
        self._line_specific_luminosity = lines
        # mimic the "nebemlineinspec" function in FSPS
        if self.params.get("nebemlineinspec", False):
            if self.ssp.params["smooth_velocity"] == True:
                dlam = self.emline_wavelengths*self.ssp.params["sigma_smooth"]/2.9979E18*1E13 #smoothing variable is in km/s
            else:
                dlam = self.ssp.params["sigma_smooth"] #smoothing variable is in AA
            nearest_id = np.searchsorted(wave, self.emline_wavelengths)
            neb_res_min = wave[nearest_id]-wave[nearest_id-1]
            dlam = np.max([dlam,neb_res_min*2], axis=0)
            gaussnebarr = [1./np.sqrt(2*np.pi)/dlam[i]*np.exp(-(wave-self.emline_wavelengths[i])**2/2/dlam[i]**2) \
            /2.9979E18*self.emline_wavelengths[i]**2 for i in range(len(lines))]
            for i in range(len(lines)):
                spec += lines[i]*gaussnebarr[i]
        return wave, spec / mtot, self.ssp.stellar_mass / mtot

    def get_galaxy_elines(self):
        """Get the wavelengths and specific emission line luminosity of the nebular emission lines
        predicted by FSPS. These lines are in units of Lsun/solar mass formed.
        This assumes that `get_galaxy_spectrum` has already been called.

        :returns ewave:
            The *restframe* wavelengths of the emission lines, AA

        :returns elum:
            Specific luminosities of the nebular emission lines,
            Lsun/stellar mass formed
        """
        
        # This allows subclasses to set their own specific emission line
        # luminosities within other methods, e.g., get_galaxy_spectrum, by
        # populating the `_specific_line_luminosity` attribute.
        elum = getattr(self, "_line_specific_luminosity", None).copy()
        ewave = self.emline_wavelengths
        
        if elum is None:
            ewave = self.ssp.emline_wavelengths
            elum = self.ssp.emline_luminosity.copy()
        if elum.ndim > 1:
            elum = elum[0]
        if self.ssp.params["sfh"] == 3:
            # tabular sfh
            mass = np.sum(self.params.get('mass', 1.0))
            elum /= mass

        return ewave, elum

    
class NebCSPBasis(CSPSpecBasis):

    """A subclass of :py:class:`SSPBasis` for combinations of N composite
    stellar populations (including single-age populations). The number of
    composite stellar populations is given by the length of the ``"mass"``
    parameter. Other population properties can also be vectors of the same
    length as ``"mass"`` if they are independent for each component.
    """

    def __init__(self, zcontinuous=1, reserved_params=['sigma_smooth'],
                 vactoair_flag=False, compute_vega_mags=False, 
                 cue_kwargs={}, **kwargs):

        # This is a StellarPopulation object from fsps
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous,
                                          vactoair_flag=vactoair_flag)
        # we do these now
        rp = ["dust1", "dust2", "dust3", "add_dust_emission",
              "add_igm_absorption", "igm_factor",
              "add_neb_emission", "add_neb_continuum", "nebemlineinspec",
              "fagn", "agn_tau"]
        reserved_params = kwargs.pop("reserved_params", []) + rp
        super().__init__(reserved_params=reserved_params, **kwargs)
        for k in ["add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"]:
            self.ssp.params[k] = False
            
        self.emul = Emulator(**cue_kwargs)
        # compile the functions first to speed up prediction, using an initial set of cue parameters in order
        _ = fast_line_prediction([19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, -2.5, 0.0, 0.0, 0.0, 49.1], self.emul)
        _ = fast_cont_prediction([19.7, 5.3, 1.6, 0.6, 3.9, 0.01, 0.2, -2.5, 0.0, 0.0, 0.0, 49.1], 
                                 self.ssp.wavelengths,
                                 self.emul, unit='Lsun/Hz')
        self.emline_wavelengths = np.genfromtxt(resource_filename("cuejax", "data/cue_emlines_info.dat"),
                                                dtype=[('wave', 'f8'), ('name', '<U20')],
                                                delimiter=',')['wave']

    def get_galaxy_spectrum(self, **params):
        """Update parameters, then loop over each component getting a spectrum
        for each and sum with appropriate weights.

        :param params:
            A parameter dictionary that gets passed to the ``self.update``
            method and will generally include physical parameters that control
            the stellar population and output spectrum or SED.

        :returns wave:
            Wavelength in angstroms.

        :returns spectrum:
            Spectrum in units of Lsun/Hz/solar masses formed.

        :returns mass_fraction:
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)
        spectra, linelum = [], []
        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)
        # Loop over mass components
        for i, m in enumerate(mass):
            self.update_component(i)
            wave, spec, lines = get_spectrum(self.ssp, self.params, self.emul, self.emline_wavelengths,
                                             tage=self.ssp.params['tage'])
            spectra.append(spec)
            mfrac[i] = (self.ssp.stellar_mass)
            linelum.append(lines)

        # Convert normalization units from per stellar mass to per mass formed
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            mass /= mfrac
        
        # mimic the "nebemlineinspec" function in FSPS
        if self.params.get("nebemlineinspec", False):
            if self.ssp.params["smooth_velocity"] == True:
                dlam = self.emline_wavelengths*self.ssp.params["sigma_smooth"]/2.9979E18*1E13 #smoothing variable is in km/s
            else:
                dlam = self.ssp.params["sigma_smooth"] #smoothing variable is in AA
            nearest_id = np.searchsorted(wave, self.emline_wavelengths)
            neb_res_min = wave[nearest_id]-wave[nearest_id-1]
            dlam = np.max([dlam,neb_res_min*2], axis=0)
            for spec_id in range(len(spectra)):
                gaussnebarr = [1./np.sqrt(2*np.pi)/dlam[i]*np.exp(-(wave-self.emline_wavelengths[i])**2/2/dlam[i]**2) \
                /2.9979E18*self.emline_wavelengths[i]**2 for i in range(len(linelum[spec_id]))]
                for i in range(len(linelum[spec_id])):
                    spectra[spec_id] += linelum[spec_id][i]*gaussnebarr[i]

        spectrum = np.dot(mass, np.array(spectra)) / mass.sum()
        self._line_specific_luminosity = np.dot(mass, np.array(linelum)) / mass.sum()
        mfrac_sum = np.dot(mass, mfrac) / mass.sum()
        return wave, spectrum, mfrac_sum  
    
    def get_galaxy_elines(self):
        """Get the wavelengths and specific emission line luminosity of the nebular emission lines
        predicted by FSPS. These lines are in units of Lsun/solar mass formed.
        This assumes that `get_galaxy_spectrum` has already been called.

        :returns ewave:
            The *restframe* wavelengths of the emission lines, AA

        :returns elum:
            Specific luminosities of the nebular emission lines,
            Lsun/stellar mass formed
        """
        
        # This allows subclasses to set their own specific emission line
        # luminosities within other methods, e.g., get_galaxy_spectrum, by
        # populating the `_specific_line_luminosity` attribute.
        elum = getattr(self, "_line_specific_luminosity", None).copy()
        ewave = self.emline_wavelengths
        
        if elum is None:
            ewave = self.ssp.emline_wavelengths
            elum = self.ssp.emline_luminosity.copy()
        if elum.ndim > 1:
            elum = elum[0]
        if self.ssp.params["sfh"] == 3:
            # tabular sfh
            mass = np.sum(self.params.get('mass', 1.0))
            elum /= mass

        return ewave, elum
    

cue_keys = ['ionspec_index1',
            'ionspec_index2',
            'ionspec_index3',
            'ionspec_index4',
            'ionspec_logLratio1',
            'ionspec_logLratio2',
            'ionspec_logLratio3',
            'gas_logu',
            'gas_lognH',
            'gas_logz',
            'gas_logno',
            'gas_logco']

def get_spectrum(ssp, params, emul, ewave, tage=0):
    """
    Add the nebular continuum from Cue to the young and old population and do the dust attenuation and igm absorption.
    Also, calculate the line luminoisities from the young csp and old csp and do the dust attenuation and igm absorption.
    The output spec/line luminosity need to be divided by total formed mass to get the specific number.
    :param use_stellar_ionizing:
        If true, fit CSPs and to get the ionizing spectrum parameters, else read from ssp
    """
    add_neb = params.get("add_neb_emission", False)
    use_stars =  params.get("use_stellar_ionizing", False)
    #params = {k: v.item() if isinstance(v, np.ndarray) else v for k, v in params.items()}
    wave, tspec = ssp.get_spectrum(tage=tage, peraa=False)
    young, old = ssp._csp_young_old
    csps = [young, old] # could combine with previous line
    lines = []
    mass = np.sum(params.get('mass', 1.0))
    if not add_neb:
        lines = [np.zeros_like(ewave), np.zeros_like(ewave)]
    elif add_neb:
        if not use_stars:
            cue_params = {k: params[k] for k in cue_keys}
            theta = np.array(list(cue_params.values()))
            line_prediction = fast_line_prediction(theta, emul)[0].squeeze() * 10**(params["gas_logqion"] - 49.1)
            #if ssp.params["sfh"] == 3:
            #    line_prediction /= mass
            lines = [line_prediction, np.zeros_like(ewave)]
            csps[0][wave>=912] += fast_cont_prediction(theta, wave[wave>=912], emul, unit='Lsun/Hz')[0].squeeze() * 10**(params["gas_logqion"] - 49.1)
        elif use_stars:
            for spec in csps:
                params.update(**fit_4loglinear_ionparam(wave, spec))
                cue_params = {k: params[k] for k in cue_keys}
                theta = np.array(list(cue_params.values()))
                line_prediction = fast_line_prediction(theta, emul)[0].squeeze() * 10**(params["gas_logqion"] - 49.1)
                lines.append(line_prediction)
                spec[wave>=912] += fast_cont_prediction(theta, wave[wave>=912], emul, unit='Lsun/Hz')[0].squeeze() * 10**(params["gas_logqion"] - 49.1)
        else:
            raise KeyError('No "use_stellar_ionizing" in model')

    sspec, lines = add_dust(wave, csps, ewave, lines, dust1_index=ssp.params['dust1_index'], **params)
    sspec = add_igm(wave, sspec, **params)
    return wave, sspec, lines


# ---------------- AGN torus emission ----------------

SPS_HOME = os.getenv('SPS_HOME')
Nenkova2008 = np.genfromtxt(
    os.path.join(SPS_HOME, 'dust', 'Nenkova08_y010_torusg_n10_q2.0.dat'),
    dtype=[('wave', 'f8'),
           ('fnu_5', '<U20'), ('fnu_10', '<U20'), ('fnu_20', '<U20'),
           ('fnu_30', '<U20'), ('fnu_40', '<U20'), ('fnu_60', '<U20'),
           ('fnu_80', '<U20'), ('fnu_100', '<U20'), ('fnu_150', '<U20')],
    delimiter='   ', skip_header=4)
agndust_tau = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0])
agndust_lam = Nenkova2008['wave']
nagndust = agndust_tau.shape[0]
nagndust_spec = Nenkova2008['wave'].shape[0]

agndust_specinit = [Nenkova2008['fnu_5'],  Nenkova2008['fnu_10'], Nenkova2008['fnu_20'],
                    Nenkova2008['fnu_30'], Nenkova2008['fnu_40'], Nenkova2008['fnu_60'],
                    Nenkova2008['fnu_80'], Nenkova2008['fnu_100'], Nenkova2008['fnu_150']]
agndust_specinit = np.vstack(agndust_specinit).T
agndust_specinit = np.array(agndust_specinit, dtype=np.float64)


def agn_torus(wave, agn_tau):
    """AGN torus emission based on Nenkova+2008.

    wave: rest-frame wavelength in Angstrom
    agn_tau: optical depth of the AGN dust torus, which affects the SED shape;
             outside (5, 150) clamps to boundary.
    """
    agndust_spec = np.zeros((len(wave), nagndust), dtype=np.float64)
    i1 = np.argmin(np.abs(wave - agndust_lam[0]))
    i2 = np.argmin(np.abs(wave - agndust_lam[nagndust_spec - 1]))
    for i in range(nagndust):
        agndust_spec[i1:i2 + 1, i] = 10**np.interp(
            np.log10(wave[i1:i2 + 1]),
            np.log10(agndust_lam),
            np.log10(agndust_specinit[:, i] + 1e-30)) - 1e-30
    linfit = interp1d(agndust_tau, agndust_spec, axis=1,
                      bounds_error=False, fill_value='extrapolate')
    return linfit(agn_tau)