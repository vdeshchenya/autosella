"""Bounded, operator-owned diagnostic artifacts; never an evaluation endpoint.

Workers upload plain JSON through the optional collector. Redis carries only a
receipt. The evaluation gateway reads this local content-addressed store without
dispatching work. The collector must share its state_root with that gateway,
not with the numerical workers.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
import threading
import time

MAINTENANCE_INTERVAL_SECONDS = 60
LOGGER = logging.getLogger(__name__)

MAX_BUNDLE_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_STATE_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_LIMIT = 64
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
EVALUATION_ID = re.compile(r"[0-9a-f]{32}\Z")
ARTIFACT_ID = re.compile(r"[0-9a-f]{64}\Z")
IDENTITY_FIELDS = {"run_id", "evaluation_id", "task_id", "source_sha256", "mol_name"}


class DiagnosticQuotaError(ValueError):
    pass


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True, allow_nan=False).encode("utf-8")


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _decode(data: bytes):
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid diagnostic JSON") from exc


def _regular_read(path: Path, limit: int) -> bytes:
    """Do not follow artifact/checkpoint symlinks or read unbounded files."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Invalid or oversized diagnostic file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Oversized diagnostic file")
        return data
    finally:
        os.close(descriptor)


def _directory(root: Path, *parts: str, create=False) -> Path:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid diagnostic state root")
    current = root
    for part in parts:
        if not isinstance(part, str) or not IDENTIFIER.fullmatch(part):
            raise ValueError("Invalid diagnostic path identity")
        current = current / part
        if current.is_symlink():
            raise ValueError("Diagnostic directory must not be a symlink")
        if create:
            current.mkdir(mode=0o700, exist_ok=True)
        if current.exists() and not current.is_dir():
            raise ValueError("Invalid diagnostic directory")
    return current


def _validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        raise ValueError("Invalid diagnostic identity")
    if not isinstance(identity["run_id"], str) or not IDENTIFIER.fullmatch(identity["run_id"]):
        raise ValueError("Invalid run_id")
    if not isinstance(identity["evaluation_id"], str) or not EVALUATION_ID.fullmatch(identity["evaluation_id"]):
        raise ValueError("Invalid evaluation_id")
    if not isinstance(identity["source_sha256"], str) or not ARTIFACT_ID.fullmatch(identity["source_sha256"]):
        raise ValueError("Invalid source_sha256")
    for key in ("task_id", "mol_name"):
        value = identity[key]
        if not isinstance(value, str) or not value or len(value) > 512 or any(ord(c) < 32 for c in value):
            raise ValueError(f"Invalid {key}")


def _job_directory(config, job, *, create=False):
    identity = {"run_id": job.get("run_id"), "evaluation_id": job.get("evaluation_id"),
                "source_sha256": job.get("source_sha256"), "task_id": "check", "mol_name": "check"}
    _validate_identity(identity)
    return _directory(config.state_root, "diagnostics", identity["run_id"],
                      identity["evaluation_id"], create=create)


def _summary(trace):
    keys = ("n_started", "n_completed", "n_steps", "n_completed_steps", "attempted", "completed",
            "attempted_calls", "completed_calls", "capture_incomplete",
            "calls_started", "calls_completed", "failure_kind", "stop_reason", "truncated")
    summary = {key: trace[key] for key in keys if key in trace
               and isinstance(trace[key], (str, int, float, bool, type(None)))}
    for key, value in list(summary.items()):
        if isinstance(value, str):
            summary[key] = value[:256]
    if isinstance(trace.get("last_attempt"), dict) and type(trace["last_attempt"].get("call_index")) is int:
        summary["last_attempted_call_index"] = trace["last_attempt"]["call_index"]
    terminal = trace.get("terminal")
    if isinstance(terminal, dict):
        for key in ("error_kind", "stop_reason", "worker_guard"):
            if isinstance(terminal.get(key), str):
                summary[key] = terminal[key][:128]
    return summary


def is_truncated(trace):
    truncation = trace.get("truncation", {})
    return bool(trace.get("truncated") or trace.get("capture_incomplete")
                or trace.get("geometry_omitted_atom_limit")
                or (isinstance(trace.get("exception"), dict) and trace["exception"].get("traceback_truncated"))
                or (isinstance(trace.get("invalid_input"), dict) and trace["invalid_input"].get("truncated"))
                or (isinstance(truncation, dict) and any(bool(v) for v in truncation.values())))


def _metadata(bundle, data):
    trace = bundle["trace"]
    truncation = is_truncated(trace)
    return {"artifact_id": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "identity": bundle["identity"], "summary": _summary(trace), "truncated": truncation}


def _validate_metadata(metadata, job, artifact_id=None):
    if not isinstance(metadata, dict):
        raise ValueError("Invalid diagnostic metadata")
    _validate_identity(metadata.get("identity"))
    for key in ("run_id", "evaluation_id", "source_sha256"):
        if metadata["identity"][key] != job[key]:
            raise ValueError("Diagnostic ownership mismatch")
    digest = metadata.get("artifact_id")
    if not isinstance(digest, str) or not ARTIFACT_ID.fullmatch(digest) or (artifact_id and digest != artifact_id):
        raise ValueError("Invalid artifact_id")
    if type(metadata.get("size_bytes")) is not int or not 0 < metadata["size_bytes"] <= MAX_BUNDLE_BYTES:
        raise ValueError("Invalid diagnostic size")
    return metadata


@contextmanager
def _run_lock(directory):
    descriptor = os.open(directory / "store.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Invalid diagnostic lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _atomic_write(path, data):
    descriptor, staging = tempfile.mkstemp(prefix=".artifact-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if os.path.exists(staging):
            os.unlink(staging)


def _ingested_at(metadata, payload):
    # Pre-retention stores have no timestamp. Reads never change payload mtime.
    timestamp = metadata.get("ingested_at", payload.stat().st_mtime)
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        raise ValueError("Invalid diagnostic ingestion time")
    return timestamp


def _store_entries(run_directory):
    """Validate the complete run before deleting anything; caller holds its lock."""
    entries = []
    for directory in run_directory.iterdir():
        if directory.name == "store.lock":
            continue
        if directory.is_symlink() or not directory.is_dir() or not EVALUATION_ID.fullmatch(directory.name):
            raise ValueError("Invalid diagnostic evaluation directory")
        pairs = {}
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError("Diagnostic artifact must not be a symlink")
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Invalid diagnostic store entry")
            if path.name.startswith(".artifact-"):
                # A process may have died between staging and atomic publication.
                entries.append((info.st_mtime, info.st_size, (path,)))
                continue
            if path.suffix not in {".json", ".meta"} or not ARTIFACT_ID.fullmatch(path.stem):
                raise ValueError("Invalid diagnostic store entry")
            pairs.setdefault(path.stem, {})[path.suffix] = path
        for digest, pair in pairs.items():
            payload, meta = pair.get(".json"), pair.get(".meta")
            timestamp = min(path.stat().st_mtime for path in pair.values())
            if meta is not None:
                metadata = _decode(_regular_read(meta, 4096))
                identity = metadata.get("identity", {}) if isinstance(metadata, dict) else {}
                job = {"run_id": run_directory.name, "evaluation_id": directory.name,
                       "source_sha256": identity.get("source_sha256")}
                _validate_metadata(metadata, job, digest)
                if payload is not None:
                    timestamp = _ingested_at(metadata, payload)
                else:
                    # A crash halfway through deletion must not leave receipts.
                    timestamp = float("-inf")
            entries.append((timestamp, sum(path.stat().st_size for path in pair.values()),
                            tuple(pair.values())))
    return sorted(entries, key=lambda entry: (entry[0], str(entry[2][0])))


def _prune(config, run_directory, *, now, reserve=0):
    """Delete paired files by age, then oldest first to make room. Lock required."""
    entries = _store_entries(run_directory)
    used = sum(entry[1] for entry in entries)
    for timestamp, size, paths in entries:
        if now - timestamp < config.diagnostics_retention_seconds and used + reserve <= config.diagnostics_max_run_bytes:
            continue
        for path in paths:
            path.unlink()
        used -= size
    # Keep the run directory and its lock inode stable for other processes.
    for directory in run_directory.iterdir():
        if directory.is_dir() and not directory.is_symlink():
            try:
                directory.rmdir()
            except OSError:
                pass  # Nonempty evaluation directory.
    return used


class ArtifactStore:
    def __init__(self, config):
        self.config = config

    def _owned_task(self, identity):
        _validate_identity(identity)
        database = Path(self.config.state_root) / "gateway.sqlite3"
        if database.is_symlink():
            raise ValueError("Invalid gateway database path")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT j.*,r.state run_state FROM jobs j JOIN runs r USING(run_id) "
                                     "WHERE evaluation_id=?", (identity["evaluation_id"],)).fetchone()
            if row is None:
                raise ValueError("Diagnostic evaluation not found")
            job = dict(row)
            release = connection.execute("SELECT value FROM gateway_metadata WHERE key='release_id'").fetchone()
            if release is None or release[0] != self.config.release_id:
                raise ValueError("Diagnostic release mismatch")
        if job["run_state"] == "revoked":
            raise ValueError("Diagnostic run is revoked")
        if any(job[key] != identity[key] for key in ("run_id", "evaluation_id", "source_sha256")):
            raise ValueError("Diagnostic job identity mismatch")
        directory = _directory(self.config.state_root, "jobs", identity["evaluation_id"])
        state = _decode(_regular_read(directory / "evaluation-state.json", MAX_STATE_BYTES))
        if not isinstance(state, dict) or state.get("task_metadata") != {
                "run_id": identity["run_id"], "evaluation_id": identity["evaluation_id"]}:
            raise ValueError("Diagnostic task ownership mismatch")
        matches = 0
        for record in state.get("records", []):
            if not isinstance(record, dict) or record.get("mol_name") != identity["mol_name"]:
                continue
            matches += sum(isinstance(attempt, dict) and attempt.get("task_id") == identity["task_id"]
                           for attempt in record.get("attempts", []))
        if matches != 1:
            raise ValueError("Diagnostic task/molecule not found in evaluation")
        return job

    def ingest(self, data: bytes):
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BUNDLE_BYTES:
            raise ValueError("Diagnostic bundle exceeds 1 MiB")
        bundle = _decode(data)
        if (not isinstance(bundle, dict) or set(bundle) != {"schema_version", "identity", "trace"}
                or type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 1
                or not isinstance(bundle.get("trace"), dict)):
            raise ValueError("Invalid diagnostic bundle schema")
        job = self._owned_task(bundle["identity"])
        metadata = _metadata(bundle, data)
        metadata["ingested_at"] = time.time()
        metadata_bytes = json_bytes(metadata)
        if len(metadata_bytes) > 4096:
            raise ValueError("Diagnostic metadata too large")
        required = len(data) + len(metadata_bytes)
        if required > self.config.diagnostics_max_run_bytes:
            raise DiagnosticQuotaError("Diagnostic bundle cannot fit run storage quota")
        run_directory = _directory(self.config.state_root, "diagnostics", job["run_id"], create=True)
        with _run_lock(run_directory):
            _prune(self.config, run_directory, now=metadata["ingested_at"])
            directory = _job_directory(self.config, job, create=True)
            target = directory / (metadata["artifact_id"] + ".json")
            meta = directory / (metadata["artifact_id"] + ".meta")
            if target.is_symlink() or meta.is_symlink():
                raise ValueError("Diagnostic artifact must not be a symlink")
            if target.exists():
                if _regular_read(target, MAX_BUNDLE_BYTES) != data:
                    raise ValueError("Immutable diagnostic artifact conflict")
                if meta.exists():
                    metadata = _validate_metadata(_decode(_regular_read(meta, 4096)), job, metadata["artifact_id"])
                else:
                    metadata["ingested_at"] = target.stat().st_mtime
                    _atomic_write(meta, json_bytes(metadata))
                return {**metadata, "availability": "stored"}
            _prune(self.config, run_directory, now=metadata["ingested_at"], reserve=required)
            directory = _job_directory(self.config, job, create=True)
            _atomic_write(target, data)
            _atomic_write(meta, metadata_bytes)
        return {**metadata, "availability": "stored"}

    def cleanup(self):
        """Sweep idle runs without requiring an upload or touching gateway state."""
        root = _directory(self.config.state_root, "diagnostics")
        if not root.exists():
            return
        for path in root.iterdir():
            try:
                directory = _directory(self.config.state_root, "diagnostics", path.name)
                with _run_lock(directory):
                    _prune(self.config, directory, now=time.time())
            except (OSError, ValueError):
                # Fail closed for a damaged run, but keep maintaining other runs.
                LOGGER.warning("Diagnostic retention sweep failed for a run", exc_info=False)



def diagnostic_manifest(config, job, molecule=None, offset=0, limit=32):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= MAX_MANIFEST_LIMIT:
        raise ValueError("Invalid diagnostic manifest page")
    if molecule is not None and (not isinstance(molecule, str) or not molecule or len(molecule) > 512):
        raise ValueError("Invalid diagnostic molecule")
    directory = _job_directory(config, job)
    entries, total = [], 0
    run_directory = directory.parent
    if run_directory.exists():
        with _run_lock(run_directory):
            for path in sorted(directory.glob("*.meta")):
                if not path.with_suffix(".json").exists():
                    continue  # Interrupted deletion; the next collector sweep repairs it.
                metadata = _validate_metadata(_decode(_regular_read(path, 4096)), job, path.stem)
                if molecule is None or metadata["identity"]["mol_name"] == molecule:
                    if offset <= total < offset + limit:
                        entries.append(metadata)
                    total += 1
    # Retention may shrink a manifest between page requests. End gracefully.
    offset = min(offset, total)
    response = {"schema_version": 1, "evaluation_id": job["evaluation_id"],
                "artifacts": entries, "total": total,
                "next_offset": min(offset + limit, total), "eof": offset + limit >= total}
    # Leave room for gateway ownership/release fields and a capture receipt.
    while len(json_bytes(response)) > MAX_RESPONSE_BYTES - 8192 and response["artifacts"]:
        response["artifacts"].pop()
        response["next_offset"] -= 1
        response["eof"] = response["next_offset"] == response["total"]
    return response


def read_diagnostic_chunk(config, job, artifact_id, offset=0, max_bytes=65536):
    if not isinstance(artifact_id, str) or not ARTIFACT_ID.fullmatch(artifact_id):
        raise ValueError("Invalid artifact_id")
    if type(offset) is not int or offset < 0 or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_RESPONSE_BYTES:
        raise ValueError("Invalid diagnostic chunk bounds")
    directory = _job_directory(config, job)
    try:
        with _run_lock(directory.parent):
            metadata = _validate_metadata(_decode(_regular_read(directory / (artifact_id + ".meta"), 4096)), job, artifact_id)
            data = _regular_read(directory / (artifact_id + ".json"), MAX_BUNDLE_BYTES)
    except FileNotFoundError as exc:
        raise ValueError("Diagnostic artifact unavailable (expired, evicted, or not uploaded)") from exc
    if len(data) != metadata["size_bytes"] or hashlib.sha256(data).hexdigest() != artifact_id:
        raise ValueError("Diagnostic artifact integrity mismatch")
    if offset > len(data):
        raise ValueError("Diagnostic offset exceeds artifact size")
    piece = data[offset:offset + min(max_bytes, 45000)]
    return {"schema_version": 1, "evaluation_id": job["evaluation_id"], "artifact_id": artifact_id,
            "identity": metadata["identity"], "size_bytes": len(data), "sha256": artifact_id,
            "offset": offset, "next_offset": offset + len(piece), "eof": offset + len(piece) == len(data),
            "encoding": "base64", "data": base64.b64encode(piece).decode("ascii"),
            "truncated": metadata["truncated"]}


def read_token(path) -> str:
    token = _regular_read(Path(path), 4096).decode("ascii").strip()
    if len(token) < 32 or any(c.isspace() for c in token):
        raise ValueError("Diagnostic token must contain at least 32 non-whitespace ASCII characters")
    return token


def collector_server(config, bind="127.0.0.1", port=0):
    """Construct the optional service; callers explicitly start serve_forever."""
    if config.diagnostics_token_file is None:
        raise ValueError("diagnostics_token_file is required to enable the collector")
    token = read_token(config.diagnostics_token_file)
    store = ArtifactStore(config)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_args):
            pass  # Never log the bearer token or diagnostic body.

        def do_POST(self):
            if self.path != "/v1/artifacts":
                self.send_error(404)
                return
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode("utf-8"), ("Bearer " + token).encode("ascii")):
                self.send_error(401)
                return
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Type") != "application/json":
                self.send_error(400)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BUNDLE_BYTES:
                    self.send_error(413)
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    self.send_error(400)
                    return
                receipt = store.ingest(body)
                encoded = json_bytes({"ok": True, "receipt": receipt})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except DiagnosticQuotaError:
                self.send_error(507, "Diagnostic quota reached")
            except (ValueError, TypeError):
                self.send_error(400, "Diagnostic bundle rejected")
            except (OSError, sqlite3.Error):
                self.send_error(503, "Diagnostic storage temporarily unavailable")

    class BoundedServer(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

        def __init__(self, *args, **kwargs):
            self.slots = threading.BoundedSemaphore(8)
            self.next_maintenance = 0.0
            super().__init__(*args, **kwargs)

        def service_actions(self):
            if time.monotonic() >= self.next_maintenance:
                self.next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL_SECONDS
                try:
                    store.cleanup()
                except (OSError, ValueError):
                    LOGGER.warning("Diagnostic retention sweep unavailable", exc_info=False)

        def process_request(self, request, client_address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.slots.release()

    return BoundedServer((bind, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args(argv)
    # Fixed-path ``python -I .../diagnostics.py`` excludes ambient imports.
    # Add only the trusted checkout containing this script, as server.py does.
    import sys
    trusted_repo = str(Path(__file__).resolve().parent.parent)
    if trusted_repo not in sys.path:
        sys.path.insert(0, trusted_repo)
    from evaluation_access.server import Config
    with collector_server(Config.load(args.config), args.bind, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
