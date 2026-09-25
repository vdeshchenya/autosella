"""algo.py is a by-value vendoring of Sella; guard the seams where it can drift.

algo.py cannot import from the repo -- it is cloudpickled to the validation
workers by value -- so it re-declares the unit constants that
minimizers/sella_wrapper.py on main imports from utils. Those copies must be the *same
doubles*, not merely the same number to printed precision.

This is not pedantry. Every energy and force handed to Sella is divided by
_EV_TO_KJ, and Sella's trust-region step is chaotic at the ULP level. A literal
that was 201 ULP off (relative 3e-14) made algo.py and the real sella package
take different numbers of force calls on 2 of 250 train molecules, which broke
the `mean_rel_energy >= 1.0` gate for the *unmodified* starting program and
would have stopped the autoresearch loop from accepting anything.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import utils

ROOT = Path(__file__).resolve().parent.parent


def _load_algo():
    spec = importlib.util.spec_from_file_location("_algo_vendoring", ROOT / "algo.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # jax / ase absent in a minimal install
        pytest.skip(f"algo.py not importable here: {exc}")
    return module


def _ulps(a: float, b: float) -> int:
    return int(abs(np.float64(a).view(np.int64) - np.float64(b).view(np.int64)))


@pytest.mark.parametrize(
    "algo_name,utils_name",
    [("_EV_TO_KJ", "EV_TO_KJ"), ("_ANGSTROM_TO_NM", "ANGSTROM_TO_NM")],
)
def test_vendored_constants_are_bit_identical_to_utils(algo_name, utils_name):
    algo = _load_algo()
    mine = getattr(algo, algo_name)
    theirs = getattr(utils, utils_name)
    assert mine == theirs, (
        f"algo.{algo_name}={mine!r} != utils.{utils_name}={theirs!r} "
        f"({_ulps(mine, theirs)} ULP apart). algo.py and sella_wrapper.py would "
        "then run on differently scaled potentials and diverge."
    )


def test_ev_to_kj_matches_the_documented_construction():
    """Pins the value itself, not just agreement between two copies that could
    drift together if someone edited both."""
    assert utils.EV_TO_KJ == 2625.4996394799 * (1.0 / 27.211386245988)
    assert utils.EV_TO_KJ == utils.HARTREE_TO_KJ * utils.EV_TO_HARTREE


def test_algo_exposes_the_harness_entrypoint():
    algo = _load_algo()
    assert callable(algo.minimize_func)
    assert callable(algo.entrypoint)
    assert algo.entrypoint() is algo.minimize_func


# ── harness-local constants ─────────────────────────────────────────────────

def test_kcal_to_kj_agrees_across_the_harness():
    """KCAL_TO_KJ is defined per-module rather than in utils.py.

    utils.py is ported verbatim from the main data-preparation source; its
    checksum is pinned below. 4.184 is exact by definition (thermochemical calorie),
    so duplicated literals carry no rounding risk -- but they still have to agree.
    """
    import validate
    from distributed_validate import worker

    assert validate.KCAL_TO_KJ == worker.KCAL_TO_KJ == 4.184


def test_utils_matches_main_preparation_source():
    """Pin the shared unit/XYZ implementation to the audited main source.

    Source: f475a971a906f945b8e987d97cf7902193ba58a8:molecules/utils.py.
    AR convergence and calculator modules adapt that source's API and failure
    handling, so their behavior is covered by focused tests instead of hashes.
    """
    import hashlib

    actual = hashlib.sha256((ROOT / "utils.py").read_bytes()).hexdigest()
    assert actual == "fd78b0578266858a455bf7087ee0b26098d5eb8edc28c6a8d5ebb3aca1be9452"
