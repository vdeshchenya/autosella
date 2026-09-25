"""Bounded diagnostic storage tests using fixed arrays, never a real calculator."""
import json

import numpy as np

from distributed_validate import diagnostics


def _capture(tmp_path, atoms=3):
    capture = diagnostics.TaskCapture(str(tmp_path), {})
    capture.initialize(np.ones(atoms, dtype=int))
    return capture


def _failure():
    return {"error": "test failure", "error_kind": "optimizer_error", "result": None}


def test_ring_retains_last_eight_frames_and_two_hundred_scalar_rows(tmp_path):
    capture = _capture(tmp_path)
    pos = np.arange(9, dtype=float).reshape(3, 3)
    for call in range(1, 206):
        capture.attempted(pos + call, call, call - 1)
        capture.completed(pos + call, -pos, float(call), call, call)
        capture.metrics(call, {"converged": 0, "max_force_eh_bohr": 1.0})
    capture.close()
    bundle = diagnostics.read_bundle(str(tmp_path), _failure())
    assert [f["call_index"] for f in bundle["frames"]] == list(range(198, 206))
    assert [r["call_index"] for r in bundle["scalar_rows"]] == list(range(6, 206))
    assert bundle["truncation"]["older_completed_frames_not_retained"] == 197
    assert bundle["truncation"]["older_scalar_rows_not_retained"] == 5
    assert bundle["attempted_calls"] == bundle["completed_calls"] == 205
    assert bundle["truncated"] is True


def test_parent_ignores_uncommitted_torn_slots(tmp_path):
    capture = _capture(tmp_path)
    pos = np.ones((3, 3))
    capture.attempted(pos, 1, 0)
    capture.completed(pos, -pos, 1.0, 1, 1)
    capture.attempted(pos * 2, 2, 1)
    # Simulate interruption while writing a new slot, before its commit marker.
    capture.arrays["positions"][1] = 999
    capture.arrays["rows"][1]["call_index"] = 0
    capture.close()
    bundle = diagnostics.read_bundle(str(tmp_path), _failure())
    assert [f["call_index"] for f in bundle["frames"]] == [1]
    assert [r["call_index"] for r in bundle["scalar_rows"]] == [1]
    assert bundle["last_attempt"]["call_index"] == 2


def test_bundle_byte_cap_drops_whole_old_frames_explicitly(tmp_path):
    atoms = diagnostics.MAX_CAPTURE_ATOMS
    capture = _capture(tmp_path, atoms)
    # Many decimal digits force an artifact beyond the JSON cap even though the
    # underlying fixed-size binary journal remains below one MiB.
    pos = np.full((atoms, 3), 0.12345678912345678)
    for call in range(1, 9):
        capture.attempted(pos, call, call - 1)
        capture.completed(pos, -pos, -123.123456789, call, call)
    capture.close()
    assert (tmp_path / "journal.bin").stat().st_size < 1024 * 1024
    bundle = diagnostics.read_bundle(str(tmp_path), _failure())
    encoded = json.dumps(bundle, allow_nan=False, separators=(",", ":")).encode()
    assert len(encoded) <= diagnostics.MAX_BUNDLE_BYTES
    assert bundle["truncation"]["frames_dropped_for_size"] > 0
    assert bundle["truncated"] is True
    assert bundle["frames"][-1]["call_index"] == 8
    assert len(bundle["frames"][-1]["positions"]) == atoms
    assert len(bundle["last_attempt"]["positions"]) == atoms


def test_oversize_molecule_omits_full_geometry_instead_of_partial_atoms(tmp_path):
    atoms = diagnostics.MAX_CAPTURE_ATOMS + 1
    capture = _capture(tmp_path, atoms)
    pos = np.ones((atoms, 3))
    capture.attempted(pos, 1, 0)
    capture.completed(pos, -pos, 1.0, 1, 1)
    capture.close()
    bundle = diagnostics.read_bundle(str(tmp_path), _failure())
    assert bundle["geometry_omitted_atom_limit"] is True
    assert bundle["truncated"] is True
    assert bundle["atom_count"] == atoms
    assert bundle["frames"] == []
    assert bundle["last_attempt"]["positions"] is None
    assert len(bundle["scalar_rows"]) == 1


def test_exception_text_is_bounded_and_first_failure_is_preserved(tmp_path):
    capture = _capture(tmp_path)
    capture.exception(ValueError("original " + "x" * 100000))
    capture.exception(RuntimeError("secondary"))
    capture.close()
    bundle = diagnostics.read_bundle(str(tmp_path), _failure())
    assert bundle["exception"]["type"] == "ValueError"
    assert bundle["exception"]["message"].startswith("original ")
    assert len(bundle["exception"]["traceback"].encode()) <= diagnostics.MAX_TEXT_BYTES
    assert bundle["exception"]["traceback_truncated"] is True


def test_missing_or_incomplete_journal_is_diagnostic_only(tmp_path):
    capture = _capture(tmp_path)
    capture.close()
    (tmp_path / "journal.bin").write_bytes(b"broken")
    bundle = diagnostics.read_bundle(str(tmp_path), _failure())
    assert bundle["capture_incomplete"] is True
    assert bundle["frames"] == []
