#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests of the nautilus_transform_in_likelihood option of run_nested_sampler.

With the option on, nautilus gets the identity as its prior and a likelihood
that applies the prior transform itself. These tests check that the log
likelihoods, the posterior points and the written chain are the same as with
the option off, that the wrapped likelihood can be pickled and used in fork
and spawn pools, and that a checkpoint can be resumed with the other setting.
They use a toy model and likelihood, so FSPS is not needed.
"""

import multiprocessing
import pickle
import shutil
from functools import partial

import h5py
import numpy as np
import pytest

from prospect.models import ProspectorParams, priors
from prospect.fitting import fit_model
from prospect.fitting.nested import (run_nested_sampler, lnlike_of_unit_cube,
                                     unit_cube_identity)

nautilus = pytest.importorskip("nautilus")

# small and fast nautilus settings, with a fixed seed
FAST = dict(nested_nlive=200, nested_neff=300, n_networks=1, seed=17)
MU = np.array([0.5, -1.0, 0.3, 0.7])
SIGMA = np.array([0.4, 0.8, 0.5, 0.3])


def build_model():
    """Return a model with four free parameters and three kinds of prior."""
    config = [
        dict(name="a", N=1, isfree=True, init=0.0,
             prior=priors.Uniform(mini=-3.0, maxi=3.0)),
        dict(name="b", N=1, isfree=True, init=0.0,
             prior=priors.Normal(mean=0.0, sigma=2.0)),
        dict(name="c", N=2, isfree=True, init=np.zeros(2),
             prior=priors.MultiVariateNormal(mean=np.zeros(2),
                                             Sigma=np.array([[1.0, 0.3],
                                                             [0.3, 0.5]]))),
    ]
    return ProspectorParams(config, verbose=False)


def toy_lnlike(theta):
    """Return a Gaussian log likelihood of the parameter vector."""
    return -0.5 * np.sum(((theta - MU) / SIGMA)**2)


def toy_lnprobfn(theta, observations=None, model=None, sps=None,
                 nested=False, **extras):
    """Return the toy log likelihood, with the prior added unless nested."""
    lnp_prior = model.prior_product(theta, nested=nested)
    if not np.isfinite(lnp_prior):
        return -np.inf
    return lnp_prior + toy_lnlike(theta)


class CountingLnlike:
    """The toy log likelihood, counting its calls in this process."""

    def __init__(self):
        self.n_call = 0

    def __call__(self, theta):
        self.n_call += 1
        return toy_lnlike(theta)


def unit_points(ndim, n=50, seed=3):
    return np.random.default_rng(seed).uniform(size=(n, ndim))


def assert_same_run(out_a, obj_a, out_b, obj_b):
    """Check two nautilus runs are bitwise the same."""
    assert len(obj_a.points) == len(obj_b.points)
    for pa, pb, la, lb in zip(obj_a.points, obj_b.points, obj_a.log_l, obj_b.log_l):
        assert np.array_equal(pa, pb)
        assert np.array_equal(la, lb)
    for key in ["points", "log_weight", "log_like"]:
        assert out_a[key].shape == out_b[key].shape
        assert np.array_equal(out_a[key], out_b[key])
    assert obj_a.log_z == obj_b.log_z


def test_unit_likelihood_value():
    model = build_model()
    lnlike = partial(lnlike_of_unit_cube, prior_transform=model.prior_transform,
                     likelihood_function=toy_lnlike)
    for u in unit_points(model.ndim):
        assert unit_cube_identity(u) is u
        assert lnlike(u) == toy_lnlike(model.prior_transform(u))


def test_unit_likelihood_pickle():
    model = build_model()
    lnlike = partial(lnlike_of_unit_cube, prior_transform=model.prior_transform,
                     likelihood_function=toy_lnlike)
    lnlike2 = pickle.loads(pickle.dumps(lnlike))
    identity2 = pickle.loads(pickle.dumps(unit_cube_identity))
    for u in unit_points(model.ndim):
        assert lnlike2(u) == lnlike(u)
        assert np.array_equal(identity2(u), u)


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_unit_likelihood_pool(method):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {method} not available")
    model = build_model()
    lnlike = partial(lnlike_of_unit_cube, prior_transform=model.prior_transform,
                     likelihood_function=toy_lnlike)
    u = unit_points(model.ndim)
    serial = np.array([lnlike(x) for x in u])
    with multiprocessing.get_context(method).Pool(2) as pool:
        pooled = np.array(pool.map(lnlike, u))
    assert np.array_equal(serial, pooled)


def test_default_matches_direct_nautilus():
    # the default path must give what nautilus gives with model.prior_transform
    model = build_model()
    out, obj = run_nested_sampler(model, toy_lnlike, nested_sampler="nautilus",
                                  **FAST)
    sampler = nautilus.Sampler(model.prior_transform, toy_lnlike, n_dim=model.ndim,
                               pass_dict=False, n_live=FAST["nested_nlive"],
                               n_networks=FAST["n_networks"], seed=FAST["seed"])
    sampler.run(n_eff=FAST["nested_neff"], verbose=False)
    points, log_w, log_l = sampler.posterior()
    direct = dict(points=points, log_weight=log_w, log_like=log_l)
    assert_same_run(out, obj, direct, sampler)


@pytest.mark.parametrize("pool", [None, "fork", "spawn", 2])
def test_transform_in_likelihood_same_result(pool):
    # pool 2 makes nautilus start its own pool with the default start method
    model = build_model()
    if isinstance(pool, str) and pool not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {pool} not available")
    runs = []
    for flag in [False, True]:
        kwargs = dict(FAST)
        if isinstance(pool, str):
            with multiprocessing.get_context(pool).Pool(2) as p:
                runs.append(run_nested_sampler(model, toy_lnlike,
                                               nested_sampler="nautilus",
                                               nautilus_transform_in_likelihood=flag,
                                               pool=p, **kwargs))
        else:
            runs.append(run_nested_sampler(model, toy_lnlike, nested_sampler="nautilus",
                                           nautilus_transform_in_likelihood=flag,
                                           pool=pool, **kwargs))
            if pool is not None:
                runs[-1][1].pool_l.pool.close()
    (out_a, obj_a), (out_b, obj_b) = runs
    assert obj_a.prior.func == model.prior_transform
    assert obj_b.prior.func is unit_cube_identity
    assert_same_run(out_a, obj_a, out_b, obj_b)
    # returned points are in parameter space, the sampler object's are not
    u, _, _ = obj_b.posterior()
    assert np.all((u >= 0) & (u <= 1))
    assert not np.all((out_b["points"] >= 0) & (out_b["points"] <= 1))
    assert np.array_equal(out_b["points"],
                          np.array([model.prior_transform(x) for x in u]))


def test_fit_model_chain(tmp_path):
    from prospect.io.write_results import write_hdf5
    from prospect.io.read_results import results_from
    model = build_model()
    chains = []
    for flag in [False, True]:
        run_params = dict(nested_sampler="nautilus", nested_nlive=200,
                          nested_target_n_effective=300, n_networks=1, seed=17,
                          nautilus_transform_in_likelihood=flag)
        out = fit_model([], model, None, lnprobfn=toy_lnprobfn, **run_params)
        hfile = str(tmp_path / f"nautilus_{flag}.h5")
        write_hdf5(hfile, run_params, model, None, out["sampling"], None)
        res, _, _ = results_from(hfile)
        chains.append(res)
    for key in ["chain", "weights", "lnprobability", "lnlike"]:
        assert np.array_equal(chains[0][key], chains[1][key])


def test_resume_with_other_setting(tmp_path):
    # runs stopped early write the same checkpoint with either setting
    model = build_model()
    n_stop = {}
    for flag in [False, True]:
        _, obj = run_nested_sampler(model, toy_lnlike, nested_sampler="nautilus",
                                    nautilus_transform_in_likelihood=flag,
                                    filepath=str(tmp_path / f"stop_{flag}.h5"),
                                    n_like_max=600, **FAST)
        n_stop[flag] = obj.n_like
    assert n_stop[False] == n_stop[True]
    with h5py.File(tmp_path / "stop_False.h5") as fa, h5py.File(tmp_path / "stop_True.h5") as fb:
        assert sorted(fa["sampler"].keys()) == sorted(fb["sampler"].keys())
        for key in fa["sampler"].keys():
            assert np.array_equal(fa["sampler"][key][...], fb["sampler"][key][...])

    # resume each checkpoint with each setting; all four runs must agree
    finished = []
    for first in [False, True]:
        for second in [False, True]:
            path = tmp_path / f"resume_{first}_{second}.h5"
            shutil.copy(tmp_path / f"stop_{first}.h5", path)
            lnlike = CountingLnlike()
            out, obj = run_nested_sampler(model, lnlike, nested_sampler="nautilus",
                                          nautilus_transform_in_likelihood=second,
                                          filepath=str(path), resume=True, **FAST)
            # only the calls after the checkpoint were made, so it was resumed
            assert obj.n_like > n_stop[first]
            assert lnlike.n_call == obj.n_like - n_stop[first]
            finished.append((out, obj))
    out_a, obj_a = finished[0]
    for out_b, obj_b in finished[1:]:
        assert_same_run(out_a, obj_a, out_b, obj_b)


def test_option_needs_nautilus():
    model = build_model()
    with pytest.raises(ValueError):
        run_nested_sampler(model, toy_lnlike, nested_sampler="dynesty",
                           nautilus_transform_in_likelihood=True)
    with pytest.raises(ValueError):
        run_nested_sampler(model, toy_lnlike, nested_sampler="nautilus",
                           nautilus_transform_in_likelihood=True, vectorized=True)
