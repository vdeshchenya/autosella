"""Passive, bounded per-task journal, read by the guard after child termination.

The child writes numeric arrays into a shared file mapping, not JSON on every
force call. Slot commit markers are cleared before writes and published last;
the parent reads only after stopping the writer. No flush/fsync is needed for
process-loss recovery on the same host (this is not a power-loss journal).
Diagnostics are best effort and must never become an optimizer stop condition.
"""
from __future__ import annotations

import hashlib
import json
import math
import mmap
from pathlib import Path
import time
import traceback

import numpy as np

FRAME_LIMIT = 8
ROW_LIMIT = 200
MAX_CAPTURE_ATOMS = 2048
# Leave space for the trusted uploader's identity envelope.
MAX_BUNDLE_BYTES = 1024 * 1024 - 8192
MAX_TEXT_BYTES = 32768
MAX_SIDECAR_BYTES = 256 * 1024
_ROW_DTYPE = np.dtype([
    ("call_index", "<i8"), ("completed_count", "<i8"), ("stage", "<i8"),
    ("elapsed_seconds", "<f8"), ("energy_kj_mol", "<f8"),
    ("energy_change_eh", "<f8"), ("max_force_eh_bohr", "<f8"),
    ("rms_force_eh_bohr", "<f8"), ("max_displacement_bohr", "<f8"),
    ("rms_displacement_bohr", "<f8"), ("converged", "<i8"),
])


def _views(buffer, atom_count: int) -> dict:
    """One deterministic layout; zero geometry atoms means scalars only."""
    result = {}
    offset = 0
    for name, shape, dtype in [
        ("header", (4,), np.dtype("<i8")),
        ("frame_ids", (FRAME_LIMIT,), np.dtype("<i8")),
        ("rows", (ROW_LIMIT,), _ROW_DTYPE),
        ("attempt", (atom_count, 3), np.dtype("<f8")),
        ("positions", (FRAME_LIMIT, atom_count, 3), np.dtype("<f8")),
        ("forces", (FRAME_LIMIT, atom_count, 3), np.dtype("<f8")),
    ]:
        if buffer is not None:
            result[name] = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
        offset += math.prod(shape) * dtype.itemsize
    return result if buffer is not None else offset


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _text(value: str, limit: int = MAX_TEXT_BYTES) -> str:
    return value.encode("utf-8", errors="replace")[:limit].decode("utf-8", errors="ignore")


def _finite(value):
    """Keep the final artifact strict JSON, marking unavailable metrics as null."""
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class TaskCapture:
    def __init__(self, directory: str, task: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.mapping = None
        self.arrays = {}
        self.disabled = False
        self.has_exception = False
        spec = task.get("optimizer_spec", {})
        source = spec.get("source") if spec.get("kind") == "source" else None
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest() if isinstance(source, str) else None
        self.metadata = {
            "schema": 1,
            "source_sha256": digest,
            "source_map": ({f"<_optimizer_source_{digest}>": digest} if digest else {}),
            "limits": {"frames": FRAME_LIMIT, "scalar_rows": ROW_LIMIT,
                       "geometry_atoms": MAX_CAPTURE_ATOMS, "bundle_bytes": MAX_BUNDLE_BYTES},
            "units": {"positions": "nm", "forces": "kJ/mol/nm", "energy": "kJ/mol",
                      "force_metrics": "Eh/Bohr", "displacement_metrics": "Bohr"},
            "nonfinite_values_encoded_as_null": True,
        }
        _atomic_json(self.directory / "metadata.json", self.metadata)

    @classmethod
    def create(cls, task: dict):
        if not task.get("_capture_diagnostics") or not task.get("_diagnostics_dir"):
            return None
        try:
            return cls(task["_diagnostics_dir"], task)
        except Exception:
            return None

    def call(self, operation: str, *args) -> None:
        """A failed diagnostic write cannot invalidate a scientific calculation."""
        if self.disabled and operation != "close":
            return
        try:
            getattr(self, operation)(*args)
        except Exception:
            self.disabled = True
            try:
                _atomic_json(self.directory / "capture_error.json", {"capture_incomplete": True})
            except Exception:
                pass

    def initialize(self, atomic_numbers: np.ndarray) -> None:
        count = len(atomic_numbers)
        retained = count if count <= MAX_CAPTURE_ATOMS else 0
        self.metadata.update({
            "atom_count": count, "geometry_atom_count": retained,
            "atomic_numbers": atomic_numbers.tolist() if retained else [],
            "geometry_omitted_atom_limit": not bool(retained),
        })
        path = self.directory / "journal.bin"
        with path.open("w+b") as stream:
            stream.truncate(_views(None, retained))
            self.mapping = mmap.mmap(stream.fileno(), 0)
        self.arrays = _views(self.mapping, retained)
        _atomic_json(self.directory / "metadata.json", self.metadata)

    def attempted(self, positions: np.ndarray, attempted: int, completed: int) -> None:
        arrays = self.arrays
        header = arrays["header"]
        header[0] = attempted
        header[1] = completed
        header[2] = 0  # latest attempted geometry is not committed yet
        if self.metadata["geometry_atom_count"]:
            arrays["attempt"][:] = positions
        header[2] = attempted
        row = arrays["rows"][(attempted - 1) % ROW_LIMIT]
        row["call_index"] = 0
        for name in _ROW_DTYPE.names:
            if _ROW_DTYPE[name].kind == "f":
                row[name] = np.nan
        row["completed_count"] = completed
        row["stage"] = 1  # backend call attempted, not yet complete
        row["elapsed_seconds"] = time.monotonic() - self.started
        row["converged"] = 0
        row["call_index"] = attempted

    def completed(self, positions: np.ndarray, forces: np.ndarray, energy: float,
                  attempted: int, completed: int) -> None:
        arrays = self.arrays
        index = (completed - 1) % FRAME_LIMIT
        arrays["frame_ids"][index] = 0
        if self.metadata["geometry_atom_count"]:
            arrays["positions"][index] = positions
            arrays["forces"][index] = forces
        arrays["frame_ids"][index] = attempted
        row = arrays["rows"][(attempted - 1) % ROW_LIMIT]
        row["call_index"] = 0
        row["energy_kj_mol"] = energy
        row["completed_count"] = completed
        row["stage"] = 2  # force result complete, bookkeeping may be pending
        row["call_index"] = attempted
        arrays["header"][1] = completed

    def metrics(self, attempted: int, metrics: dict) -> None:
        row = self.arrays["rows"][(attempted - 1) % ROW_LIMIT]
        row["call_index"] = 0
        for name, value in metrics.items():
            if name in _ROW_DTYPE.names:
                row[name] = value if value is not None else np.nan
        row["stage"] = 3
        row["call_index"] = attempted

    def invalid_input(self, positions: np.ndarray | None) -> None:
        # Use only the already converted array: never invoke candidate conversion
        # methods a second time just for diagnostics.
        record = {"backend_call_started": False, "reason": "input_validation_failed"}
        if positions is not None:
            record["shape"] = list(positions.shape)
            limit = MAX_CAPTURE_ATOMS * 3
            record["values_flat"] = _finite(positions.flat[:limit].tolist())
            record["truncated"] = positions.size > limit
        _atomic_json(self.directory / "invalid_input.json", record)

    def exception(self, exc: BaseException) -> None:
        if self.has_exception:
            return
        self.has_exception = True
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=32))
        # invalidate() may receive a newly wrapped exception before it is raised.
        # The currently handled backend exception still carries its real frames.
        if exc.__traceback__ is None:
            active = traceback.format_exc(limit=32)
            if active != "NoneType: None\n":
                rendered = active + "\nRecorded failure: " + rendered
        _atomic_json(self.directory / "exception.json", {
            "type": type(exc).__name__, "message": _text(str(exc), 8192),
            "message_truncated": len(str(exc).encode("utf-8")) > 8192,
            "traceback": _text(rendered),
            "traceback_truncated": len(rendered.encode("utf-8")) > MAX_TEXT_BYTES,
        })

    def close(self) -> None:
        self.arrays.clear()
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None


def _read_json(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_SIDECAR_BYTES + 1)
        if len(raw) > MAX_SIDECAR_BYTES:
            return {}
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_bundle(directory: str, payload: dict) -> dict | None:
    """Called by the trusted guard only after the child is stopped/finished."""
    result = payload.get("result")
    if not payload.get("error") and result and result.get("converged"):
        return None
    path = Path(directory)
    bundle = _read_json(path / "metadata.json")
    if not bundle:
        return None
    bundle.update({"attempted_calls": 0, "completed_calls": 0, "frames": [],
                   "scalar_rows": [], "last_attempt": None,
                   "terminal": {"error_kind": payload.get("error_kind"),
                                "error": _text(str(payload.get("error") or ""), 8192),
                                "stop_reason": result.get("stop_reason") if result else None,
                                "worker_guard": payload.get("worker_guard", {}).get("kind")},
                   "exception": _read_json(path / "exception.json"),
                   "invalid_input": _read_json(path / "invalid_input.json"),
                   "truncation": {"frames_dropped_for_size": 0, "scalar_rows_dropped_for_size": 0}})
    bundle.update(_read_json(path / "capture_error.json"))
    try:
        atoms = int(bundle.get("geometry_atom_count", 0))
        if not 0 <= atoms <= MAX_CAPTURE_ATOMS:
            raise ValueError("invalid geometry atom count")
        with (path / "journal.bin").open("rb") as stream:
            if stream.seek(0, 2) != _views(None, atoms):
                raise ValueError("incomplete journal")
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                arrays = _views(mapped, atoms)
                attempted, completed, attempt_id, _ = map(int, arrays["header"])
                bundle.update(attempted_calls=attempted, completed_calls=completed)
                rows = []
                for row in arrays["rows"]:
                    if row["call_index"] > 0:
                        rows.append({name: row[name].item() for name in _ROW_DTYPE.names})
                bundle["scalar_rows"] = sorted(rows, key=lambda row: row["call_index"])
                for index, call_id in enumerate(arrays["frame_ids"]):
                    if call_id > 0 and atoms:
                        bundle["frames"].append({"call_index": int(call_id),
                            "positions": arrays["positions"][index].tolist(),
                            "forces": arrays["forces"][index].tolist()})
                bundle["frames"].sort(key=lambda frame: frame["call_index"])
                if attempt_id > 0:
                    bundle["last_attempt"] = {"call_index": attempt_id,
                        "positions": arrays["attempt"].tolist() if atoms else None}
                arrays.clear()
        bundle["truncation"].update({
            "older_completed_frames_not_retained": max(0, completed - FRAME_LIMIT),
            "older_scalar_rows_not_retained": max(0, attempted - ROW_LIMIT),
        })
    except (OSError, ValueError, BufferError):
        bundle["capture_incomplete"] = True
    bundle = _finite(bundle)
    bundle["truncated"] = False
    # Drop whole frames, never silently emit a partial molecular geometry.
    def size():
        return len(json.dumps(bundle, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    while size() > MAX_BUNDLE_BYTES and bundle["frames"]:
        bundle["frames"].pop(0)
        bundle["truncation"]["frames_dropped_for_size"] += 1
    while size() > MAX_BUNDLE_BYTES and bundle["scalar_rows"]:
        bundle["scalar_rows"].pop(0)
        bundle["truncation"]["scalar_rows_dropped_for_size"] += 1
    if size() > MAX_BUNDLE_BYTES:
        bundle["last_attempt"] = None
        bundle["invalid_input"] = {}
        bundle["truncation"]["attempted_input_omitted_for_size"] = True
    if size() > MAX_BUNDLE_BYTES:
        return {"schema": 1, "capture_incomplete": True,
                "truncated": True, "truncation": {"bundle_omitted_for_size": True}}
    # Ordinary bounded-ring eviction is also truncation, with its own explicit
    # counts; it is distinguishable from additional serialized-size omissions.
    bundle["truncated"] = bool(
        any(bundle["truncation"].values())
        or bundle.get("geometry_omitted_atom_limit")
        or bundle.get("capture_incomplete")
        or bundle["exception"].get("traceback_truncated")
        or bundle["exception"].get("message_truncated")
        or bundle["invalid_input"].get("truncated")
    )
    return bundle


def attach_bundle(task: dict, payload: dict) -> dict:
    if task.get("_capture_diagnostics") and task.get("_diagnostics_dir"):
        try:
            bundle = read_bundle(task["_diagnostics_dir"], payload)
            if bundle is not None:
                payload["_diagnostic_bundle"] = bundle
        except Exception:
            pass  # Collection failure has no effect on the scientific result.
    return payload
