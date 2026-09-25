"""Shared pytest configuration.

Auto-skips backend-dependent tests when the corresponding library can't
be imported, so the same `pytest` invocation works locally (where
xtb may be missing) and on the full-env server.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest

# Make the repo root importable (tests live one level deeper)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Make tests/fixtures importable by bare module name, so optimizer-spec tests can
# exercise the module-path load branch (`{"module_name": "sample_optimizer"}`)
# the same way a worker would.
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))


def _import_ok(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


_AVAILABLE = {
    "requires_xtb": shutil.which(os.environ.get("XTB_BIN", "xtb")) is not None,
    "requires_fakeredis": _import_ok("fakeredis"),
}


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests whose marker declares a missing dependency."""
    for item in items:
        for marker_name, available in _AVAILABLE.items():
            if marker_name in item.keywords and not available:
                item.add_marker(pytest.mark.skip(reason=f"{marker_name} not available"))


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def water_xyz_path() -> Path:
    """Path to the handwritten 3-atom H2O fixture."""
    return FIXTURES / "water.xyz"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def molecules_layout_dir() -> Path:
    """Mini `molecules/` layout used by discovery tests."""
    return FIXTURES / "molecules_layout"
