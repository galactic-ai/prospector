"""ProcessLikelihood: each process builds the model and SPS itself, so a pool never pickles them.

The toy problem of test_pocomc.py, with an SPS stand-in that cannot be pickled. FSPS is not needed.
"""

import multiprocessing as mp
import os
import pickle
import threading

import numpy as np
import pytest

from prospect.fitting import fit_model, fitting
from prospect.fitting.fitting import ProcessLikelihood

from test_pocomc import analytic, build_model, toy_lnlike, toy_lnprobfn

BUILDS = []  # process ids of the builds in this process


class LockedSPS:
    """An SPS stand-in that cannot be pickled, holding an 8 MB array."""

    def __init__(self):
        self.lock = threading.Lock()
        self.big = np.zeros(10**6)


def build_toy():
    """Return the toy observations, model and SPS, and record the build."""
    BUILDS.append(os.getpid())
    return [], build_model(), LockedSPS()


def checked_lnprobfn(theta, observations=None, model=None, sps=None, nested=False, **extras):
    """Return the toy log probability, checking that the built SPS was passed."""
    assert isinstance(sps, LockedSPS)
    return toy_lnprobfn(theta, model=model, nested=nested)


@pytest.fixture(autouse=True)
def fresh_builds():
    """Start each test with no builds in this process."""
    fitting._PROCESS_BUILDS.clear()
    BUILDS.clear()


def posterior_mean_ok(out):
    """Return True if the weighted posterior mean is within 0.15 sd of the closed form."""
    mean, sd, _ = analytic()
    w = np.exp(out["log_weight"] - out["log_weight"].max())
    got = np.average(out["points"], axis=0, weights=w)
    return bool(np.all(np.abs(got - mean) < 0.15 * sd))


def test_builds_once_and_pickles_small():
    """The pickle holds the builder only; one build per process, also after a round trip."""
    with pytest.raises(TypeError):
        pickle.dumps(LockedSPS())
    lik = ProcessLikelihood(build_toy, checked_lnprobfn, labels=build_model().theta_labels(),
                            nested=True)
    data = pickle.dumps(lik)
    assert len(data) < 1000
    theta = np.array([0.1, 0.2])
    assert lik(theta) == toy_lnlike(theta)
    assert pickle.loads(data)(theta) == toy_lnlike(theta)
    assert lik.of_unit_cube()(np.array([0.5, 0.5])) == toy_lnlike(np.zeros(2))
    assert BUILDS == [os.getpid()]


def test_label_mismatch_raises():
    """A built model with other parameters than the main one raises on the first call."""
    lik = ProcessLikelihood(build_toy, checked_lnprobfn, labels=["a", "c"], nested=True)
    with pytest.raises(ValueError, match="parameters"):
        lik(np.zeros(2))


def test_fit_model_refuses_builder_outside_nested():
    """emcee and the optimization do not take a likelihood builder."""
    with pytest.raises(ValueError, match="likelihood_builder"):
        fit_model([], build_model(), None, emcee=True, likelihood_builder=build_toy)


def test_pocomc_with_pool():
    """pocoMC through fit_model with a fork pool: only the workers build, posterior as serial."""
    run = dict(nested_sampler="pocomc", nested_target_n_effective=512, n_effective=256,
               n_active=128, random_state=0, lnprobfn=checked_lnprobfn,
               likelihood_builder=build_toy)
    with mp.get_context("fork").Pool(2) as pool:
        out = fit_model([], build_model(), None, pool=pool, **run)["sampling"]
    assert BUILDS == []
    assert posterior_mean_ok(out)
    serial = fit_model([], build_model(), None, **run)["sampling"]
    assert np.array_equal(out["points"], serial["points"])


def test_nautilus_with_pool():
    """nautilus with the prior transform in the likelihood and a fork pool."""
    pytest.importorskip("nautilus")
    run = dict(nested_sampler="nautilus", nested_nlive=500, nested_target_n_effective=2000,
               nautilus_transform_in_likelihood=True, seed=0, lnprobfn=checked_lnprobfn,
               likelihood_builder=build_toy)
    with mp.get_context("fork").Pool(2) as pool:
        out = fit_model([], build_model(), None, pool=pool, **run)["sampling"]
    assert BUILDS == []
    assert posterior_mean_ok(out)
