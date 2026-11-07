"""pocoMC through run_nested_sampler and fit_model, on a toy model with Normal priors.

The posterior and evidence of a Gaussian likelihood under Normal priors are known in closed form.
The state-file test uses a likelihood that cannot be pickled, as prospect's (which holds the model,
observations and SPS) cannot be in practice. FSPS is not needed.
"""

import glob
import threading

import numpy as np
import pytest
from scipy.stats import norm

from prospect.fitting import fit_model
from prospect.fitting.nested import run_nested_sampler
from prospect.models import ProspectorParams, priors

pytest.importorskip("pocomc")

MU, SIGMA = np.array([0.5, -1.0]), np.array([0.4, 0.8])  # likelihood
S0 = np.array([1.0, 2.0])  # prior widths, prior means zero
FAST = dict(nested_neff=512, n_effective=256, n_active=128, random_state=0)


def build_model():
    """Return a model with two free parameters under Normal priors."""
    config = [dict(name=n, N=1, isfree=True, init=0.0, prior=priors.Normal(mean=0.0, sigma=s))
              for n, s in zip("ab", S0)]
    return ProspectorParams(config, verbose=False)


def toy_lnlike(theta):
    """Return an unnormalised Gaussian log likelihood."""
    return -0.5 * np.sum(((theta - MU) / SIGMA) ** 2)


def toy_lnprobfn(theta, observations=None, model=None, sps=None, nested=False, **extras):
    """Return the toy log likelihood, with the prior added unless nested."""
    return toy_lnlike(theta) + (0.0 if nested else model.prior_product(theta))


class LockedLnlike:
    """The toy log likelihood holding a lock, so it cannot be pickled."""

    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, theta):
        return toy_lnlike(theta)


def analytic():
    """Return the posterior mean and sd and ln Z of the toy problem."""
    var = 1 / (1 / S0**2 + 1 / SIGMA**2)
    mean = var * MU / SIGMA**2
    log_z = np.sum(np.log(SIGMA * np.sqrt(2 * np.pi)) + norm.logpdf(MU, 0.0, np.sqrt(S0**2 + SIGMA**2)))
    return mean, np.sqrt(var), log_z


def test_posterior_and_evidence():
    """Weighted points and ln Z match the closed form."""
    model = build_model()
    out, obj = run_nested_sampler(model, toy_lnlike, nested_sampler="pocomc", **FAST)
    mean, sd, log_z = analytic()
    w = np.exp(out["log_weight"] - out["log_weight"].max())
    got = np.average(out["points"], axis=0, weights=w)
    assert np.all(np.abs(got - mean) < 0.15 * sd)  # about 4 times 1/sqrt(ESS), ESS ~ 600
    assert np.allclose(out["log_like"], [toy_lnlike(p) for p in out["points"]])
    log_z_got, log_z_err = obj.evidence()
    assert abs(log_z_got - log_z) < 3 * log_z_err


def test_state_without_likelihood(tmp_path):
    """States save with a likelihood that cannot be pickled, and a run resumes from one."""
    model = build_model()
    kw = dict(FAST, output_dir=str(tmp_path), output_label="pmc", save_every=2)
    run_nested_sampler(model, LockedLnlike(), nested_sampler="pocomc", **kw)
    states = sorted(glob.glob(str(tmp_path / "pmc_*.state")))
    assert len(states) >= 2
    out, obj = run_nested_sampler(model, LockedLnlike(), nested_sampler="pocomc",
                                  resume_state_path=states[0], **kw)
    _, _, log_z = analytic()
    log_z_got, log_z_err = obj.evidence()
    assert abs(log_z_got - log_z) < 3 * log_z_err


def test_fit_model_chain(tmp_path):
    """fit_model with pocomc writes a weighted chain that read_results reads back."""
    from prospect.io.read_results import results_from
    from prospect.io.write_results import write_hdf5

    model = build_model()
    run_params = dict(nested_sampler="pocomc", nested_target_n_effective=512, n_effective=256,
                      n_active=128, random_state=0)
    out = fit_model([], model, None, lnprobfn=toy_lnprobfn, **run_params)
    hfile = str(tmp_path / "pocomc.h5")
    write_hdf5(hfile, run_params, model, None, out["sampling"], None)
    res, _, _ = results_from(hfile)
    assert res["chain"].shape[0] == len(out["sampling"]["points"])
    assert np.all(np.isfinite(res["lnlike"])) and np.all(res["weights"] >= 0)
