"""
Tests for the frac_obrun / frac_nodust / dust4 port into fake_fsps.add_dust
"""
import re
from pathlib import Path
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "prospect" / "sources" / "fake_fsps.py"


def _load():
    txt = SRC.read_text()
    ns = {"np": np}
    for name in ("attenuate", "add_dust"):
        m = re.search(r"\ndef " + name + r"\(.*?(?=\ndef |\nclass )", txt, re.S)
        exec(m.group(0), ns)
    return ns["add_dust"]


add_dust = _load()

WAVE = np.array([1500., 2175., 3000., 5500., 7000., 9000.])
LWAVE = np.array([1216., 4861., 5007., 6563.])
YOUNG = np.array([1., 1., 1., 1., 1., 1.]);   OLD = np.array([2., 2., 2., 2., 2., 2.])
LY = np.array([3., 4., 5., 6.]);              LO = np.array([.5, .6, .7, .8])
SPECS = np.vstack([YOUNG, OLD]);              LINES = np.vstack([LY, LO])

DI, D2, D1, D1I = -0.7, 0.3, 0.5, -1.0   # dust_index, dust2, dust1, dust1_index


def _oracle(lam, s0, s1, fro, fnd, d4t, d4i, d4):
    """Independent closed form of main's add_dust for dust_type=0."""
    dust1_ext = np.exp(-D1 * (lam / 5500.) ** D1I)
    dust2_ext = np.exp(-((lam / 5500.) ** DI) * D2)
    dust4_ext = np.exp(-((lam / 5500.) ** d4i) * d4) if d4t != 0 else np.ones_like(lam)
    diff = dust2_ext * dust4_ext
    cspi = s0 * dust1_ext * (1 - fro) + s0 * fro + s1
    return cspi * ((1 - fnd) * diff + fnd)


def _call(fro, fnd, d4t, d4i, d4):
    return add_dust(WAVE, SPECS, LWAVE, LINES, dust_type=0, dust_index=DI, dust2=D2,
                    dust1_index=D1I, dust1=D1, frac_obrun=fro, frac_nodust=fnd,
                    dust4_type=d4t, dust4_index=d4i, dust4=d4)


CASES = [
    (0.0, 0.0, 0, 0.0, 0.0),   # backward compatible: nothing new switched on
    (0.3, 0.0, 0, 0.0, 0.0),   # frac_obrun only
    (0.0, 0.4, 0, 0.0, 0.0),   # frac_nodust only
    (0.0, 0.0, 1, -0.7, 0.5),  # dust4 only
    (0.7, 0.5, 1, -1.2, 0.8),  # all three together
    (1.0, 0.0, 0, 0.0, 0.0),   # limit: all young escapes dust1
    (0.0, 1.0, 0, 0.0, 0.0),   # limit: everything escapes diffuse dust
]


@pytest.mark.parametrize("fro,fnd,d4t,d4i,d4", CASES)
def test_matches_main_formula(fro, fnd, d4t, d4i, d4):
    spec, line = _call(fro, fnd, d4t, d4i, d4)
    assert np.allclose(spec, _oracle(WAVE, YOUNG, OLD, fro, fnd, d4t, d4i, d4))
    assert np.allclose(line, _oracle(LWAVE, LY, LO, fro, fnd, d4t, d4i, d4))


def test_backward_compatible_default_kwargs():
    """No new kwargs passed at all == the old two-component behaviour."""
    spec, line = add_dust(WAVE, SPECS, LWAVE, LINES, dust_type=0, dust_index=DI,
                          dust2=D2, dust1_index=D1I, dust1=D1)
    assert np.allclose(spec, _oracle(WAVE, YOUNG, OLD, 0, 0, 0, 0, 0))
    assert np.allclose(line, _oracle(LWAVE, LY, LO, 0, 0, 0, 0, 0))


def test_dust4_off_when_type_zero():
    """dust4_index/dust4 must be inert unless dust4_type != 0 (matches FSPS)."""
    a, _ = _call(0.0, 0.0, 0, -1.5, 9.9)   # dust4 params set but type 0
    b, _ = _call(0.0, 0.0, 0, 0.0, 0.0)
    assert np.allclose(a, b)


@pytest.mark.parametrize("dust_type", [0, 2, 4])
def test_structure_reduces_to_baseline_all_curves(dust_type):
    """With new params 0, output == s0*dust1*diffuse + s1*diffuse for every curve
    law (checks the structural rewrite didn't break non-power-law dust types)."""
    add = add_dust
    # trusted transmissions from the code's own attenuate (curve laws unchanged)
    ns = {"np": np}
    txt = SRC.read_text()
    exec(re.search(r"\ndef attenuate\(.*?(?=\ndef |\nclass )", txt, re.S).group(0), ns)
    att = ns["attenuate"]
    one = np.ones_like(WAVE)
    d1e, _ = att(one, WAVE, dust_type=dust_type, dust_index=DI, dust2=0.0, dust1_index=D1I, dust1=D1)
    dfe, _ = att(one, WAVE, dust_type=dust_type, dust_index=DI, dust2=D2, dust1_index=D1I, dust1=0.0)
    spec, _ = add(WAVE, SPECS, LWAVE, LINES, dust_type=dust_type, dust_index=DI, dust2=D2,
                  dust1_index=D1I, dust1=D1)
    assert np.allclose(spec, YOUNG * d1e * dfe + OLD * dfe)
