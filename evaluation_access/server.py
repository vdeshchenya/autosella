"""Trusted forced-command endpoint. Candidate source is data in this process."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import fcntl
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import stat
import subprocess
import sys
import time
from typing import Callable
import uuid

# Also supports a fixed absolute script path in authorized_keys under python -I.
TRUSTED_REPO = Path(__file__).resolve().parent.parent
if str(TRUSTED_REPO) not in sys.path:
    sys.path.insert(0, str(TRUSTED_REPO))

from distributed_validate.optimizer import normalize_optimizer_spec
from evaluation_access.release_policy import check_release_supported

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
EVALUATION_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_PROGRESS_BYTES = 8 * 1024 * 1024
MAX_DIAGNOSTIC_STATE_BYTES = 32 * 1024 * 1024
PROGRESS_STALE_SECONDS = 300
RUN_STATES = {"enabled", "paused", "revoked"}


class RequestError(ValueError):
    pass


def identifier(value, label="identifier") -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise RequestError(f"Invalid {label}")
    return value


def _progress_counts(state, job):
    if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
        raise ValueError("invalid_state_version")
    if state.get("task_metadata") != {"run_id": job["run_id"], "evaluation_id": job["evaluation_id"]}:
        raise ValueError("ownership_mismatch")
    records = state.get("records")
    if not isinstance(records, list):
        raise ValueError("invalid_records")
    completed, attempts, retries = 0, 0, 0
    names, task_ids = set(), set()
    for record in records:
        if not isinstance(record, dict) or "payload" not in record:
            raise ValueError("invalid_record")
        name = record.get("mol_name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("invalid_molecule_identity")
        names.add(name)
        record_attempts = record.get("attempts")
        if not isinstance(record_attempts, list) or not record_attempts:
            raise ValueError("invalid_attempts")
        for attempt in record_attempts:
            if not isinstance(attempt, dict):
                raise ValueError("invalid_attempt")
            task_id = attempt.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in task_ids:
                raise ValueError("invalid_task_identity")
            task_ids.add(task_id)
            if not isinstance(attempt.get("state"), str) or not attempt["state"]:
                raise ValueError("invalid_attempt_state")
        attempts += len(record_attempts)
        retries += len(record_attempts) - 1
        payload = record["payload"]
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise ValueError("invalid_terminal_payload")
        result, error, kind = payload.get("result"), payload.get("error"), payload.get("error_kind")
        success = isinstance(result, dict) and result.get("mol_name") == name and error is None and kind is None
        failure = result is None and isinstance(error, str) and bool(error) and kind in (None, "optimizer_error")
        if not success and not failure:
            raise ValueError("invalid_terminal_payload")
        completed += 1
    return {"total": len(records), "completed": completed, "pending": len(records) - completed,
            "attempts": attempts, "infrastructure_retries": retries}


@dataclass(frozen=True)
class Config:
    state_root: Path
    molecules_dir: Path
    release_id: str
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    allowed_splits: tuple[str, ...] = ("train", "valid")
    max_source_bytes: int = 1024 * 1024
    task_timeout_seconds: float = 60.0
    authorized_keys: Path | None = None
    diagnostics_max_run_bytes: int = 1024 * 1024 * 1024
    diagnostics_token_file: Path | None = None
    diagnostics_retention_seconds: int = 12 * 60 * 60

    def __post_init__(self):
        check_release_supported(self.release_id)

    @classmethod
    def load(cls, path: Path) -> "Config":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Gateway config must be a JSON object")
        for name in ("state_root", "molecules_dir"):
            value = Path(data[name])
            if not value.is_absolute():
                raise ValueError(f"{name} must be an absolute trusted path")
            data[name] = value.resolve()
        if data.get("authorized_keys") is not None:
            path_value = Path(data["authorized_keys"])
            if not path_value.is_absolute():
                raise ValueError("authorized_keys must be an absolute trusted path")
            data["authorized_keys"] = path_value
        if data.get("diagnostics_token_file") is not None:
            path_value = Path(data["diagnostics_token_file"])
            if not path_value.is_absolute():
                raise ValueError("diagnostics_token_file must be an absolute trusted path")
            data["diagnostics_token_file"] = path_value
        if "allowed_splits" in data:
            data["allowed_splits"] = tuple(data["allowed_splits"])
        config = cls(**data)
        if not config.allowed_splits or any(s not in {"train", "valid", "test", "dipeptides"}
                                            for s in config.allowed_splits):
            raise ValueError("Invalid allowed_splits")
        if not 0 < config.max_source_bytes <= 1024 * 1024:
            raise ValueError("max_source_bytes must be in 1..1048576")
        if not 0 < config.redis_port < 65536 or config.redis_db < 0:
            raise ValueError("Invalid Redis configuration")
        if not 0 < config.task_timeout_seconds <= 86400:
            raise ValueError("Invalid task_timeout_seconds")
        if type(config.diagnostics_max_run_bytes) is not int or config.diagnostics_max_run_bytes < 1024 * 1024:
            raise ValueError("diagnostics_max_run_bytes must be an integer of at least 1 MiB")
        if type(config.diagnostics_retention_seconds) is not int or config.diagnostics_retention_seconds <= 0:
            raise ValueError("diagnostics_retention_seconds must be a positive integer")
        identifier(config.release_id, "release_id")
        release = json.loads((config.molecules_dir / "release.json").read_text())
        if not isinstance(release, dict) or release.get("dataset_release_id") != config.release_id:
            raise ValueError("Configured release_id does not match the installed dataset release")
        return config


class Gateway:
    def __init__(self, config: Config, *, config_path: Path | None = None,
                 launch: Callable[[str], None] | None = None,
                 cancel_tasks: Callable[[dict, Path], None] | None = None):
        self.config = config
        self.config_path = config_path
        self.root = config.state_root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.root / "jobs").mkdir(exist_ok=True, mode=0o700)
        self.db_path = self.root / "gateway.sqlite3"
        self.launch = launch or self._launch
        self.cancel_tasks = cancel_tasks or self._cancel_tasks
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                    key_sha256 TEXT NOT NULL UNIQUE, created REAL NOT NULL,
                    authorized_key TEXT);
                CREATE TABLE IF NOT EXISTS jobs (
                    evaluation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    request_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    source TEXT NOT NULL, split TEXT NOT NULL,
                    candidate_commit TEXT NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL,
                    updated REAL NOT NULL, result TEXT, error TEXT,
                    UNIQUE(run_id, request_id));
                CREATE TABLE IF NOT EXISTS gateway_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            db.execute("BEGIN IMMEDIATE")
            release = db.execute("SELECT value FROM gateway_metadata WHERE key='release_id'").fetchone()
            if release is None:
                if db.execute("SELECT 1 FROM jobs LIMIT 1").fetchone() is not None:
                    raise RequestError("Legacy gateway state lacks a dataset release binding; use a new state_root")
                db.execute("INSERT INTO gateway_metadata VALUES ('release_id', ?)", (config.release_id,))
            elif release["value"] != config.release_id:
                raise RequestError("Gateway state belongs to a different dataset release; use a new state_root")
        os.chmod(self.db_path, 0o600)

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def job_dir(self, evaluation_id: str) -> Path:
        if not isinstance(evaluation_id, str) or not EVALUATION_ID.fullmatch(evaluation_id):
            raise RequestError("Invalid evaluation_id")
        path = self.root / "jobs" / evaluation_id
        if path.is_symlink():
            raise RequestError("Unsafe job path")
        return path

    def provision(self, run_id: str, public_key: str):
        identifier(run_id, "run_id")
        fields = public_key.strip().split()
        if len(fields) < 2 or fields[0] != "ssh-ed25519":
            raise RequestError("An ssh-ed25519 public key is required")
        try:
            key = base64.b64decode(fields[1], validate=True)
        except ValueError as exc:
            raise RequestError("Invalid public key") from exc
        if len(key) != 51 or key[:19] != b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20":
            raise RequestError("Invalid Ed25519 key encoding")
        digest = hashlib.sha256(key).hexdigest()
        with self.db() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row and row["key_sha256"] != digest:
                raise RequestError("Run is already bound to a different key; use a new run_id")
            if not row:
                if db.execute("SELECT 1 FROM runs WHERE key_sha256=?", (digest,)).fetchone():
                    raise RequestError("Public key already belongs to another run")
                db.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (run_id, "enabled", digest, time.time(), None))
        if self.config_path is None:
            raise ValueError("Provision requires fixed config_path")
        command = shlex.join([sys.executable, "-I", str(Path(__file__).resolve()), "serve",
                              "--config", str(self.config_path.resolve()), "--run-id", run_id])
        command = command.replace("\\", "\\\\").replace('"', '\\"')
        line = f'restrict,command="{command}" {fields[0]} {fields[1]} evaluation-run:{run_id}'
        with self.db() as db:
            db.execute("UPDATE runs SET authorized_key=? WHERE run_id=?", (line, run_id))
            state = db.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()["state"]
        if self.config.authorized_keys and state != "revoked":
            self._install_key(run_id, line)
        return line

    def _install_key(self, run_id, line):
        path = self.config.authorized_keys
        if path is None:
            return
        if path.is_symlink():
            raise RequestError("authorized_keys must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / "authorized-keys.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lines = path.read_text().splitlines() if path.exists() else []
            marker = f" evaluation-run:{run_id}"
            lines = [entry for entry in lines if not entry.endswith(marker)]
            if line:
                lines.append(line)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8") as output:
                    os.chmod(temporary, 0o600)
                    output.write("\n".join(lines) + ("\n" if lines else ""))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def _authorize(self, db, run_id: str, *, write=False):
        identifier(run_id, "run_id")
        run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not run or run["state"] == "revoked":
            raise RequestError("Run access is unavailable")
        if write and run["state"] != "enabled":
            raise RequestError("Run is paused")
        return run

    def _owned(self, db, run_id: str, evaluation_id: str):
        self.job_dir(evaluation_id)
        row = db.execute("SELECT * FROM jobs WHERE evaluation_id=? AND run_id=?",
                         (evaluation_id, run_id)).fetchone()
        if row is None:
            raise RequestError("Evaluation not found")
        return dict(row)

    def set_run_state(self, run_id: str, state: str):
        identifier(run_id, "run_id")
        if state not in RUN_STATES:
            raise RequestError("Invalid run state")
        with self.db() as db:
            if not db.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
                raise RequestError("Run not found")
            db.execute("UPDATE runs SET state=? WHERE run_id=?", (state, run_id))
            key_line = db.execute("SELECT authorized_key FROM runs WHERE run_id=?", (run_id,)).fetchone()["authorized_key"]
            jobs = [dict(r) for r in db.execute(
                "SELECT * FROM jobs WHERE run_id=? AND status != 'complete'", (run_id,))]
            if state != "enabled":
                db.execute("UPDATE jobs SET status='cancelling', updated=? WHERE run_id=? AND status != 'complete'",
                           (time.time(), run_id))
        if state != "enabled":
            for job in jobs:
                self._finish_cancel(job)
        self._install_key(run_id, None if state == "revoked" else key_line)
        return {"run_id": run_id, "state": state}

    def run_status(self, run_id):
        identifier(run_id, "run_id")
        with self.db() as db:
            row = db.execute("SELECT run_id,state,created FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise RequestError("Run not found")
            result = dict(row)
            result["evaluations"] = [self._public(dict(r)) for r in db.execute(
                "SELECT * FROM jobs WHERE run_id=? ORDER BY created", (run_id,))]
            return result

    def _public(self, job: dict, include_results=False):
        value = {k: job[k] for k in ("evaluation_id", "run_id", "source_sha256", "split", "status")}
        value.update(program_id=job["source_sha256"], release_id=self.config.release_id,
                     dataset_release_id=self.config.release_id, candidate_commit=job["candidate_commit"],
                     timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(job["created"])))
        if include_results and job["result"]:
            value.update(json.loads(job["result"]))
            value["status"] = job["status"]
        if job["error"]:
            value["infrastructure_error"] = job["error"]
        value["progress"] = self._progress(job)
        return value

    def _progress(self, job: dict):
        """Project an owned durable snapshot without Redis, locks, or writes.

        Completed counts terminal molecule outcomes, including candidate errors;
        it never represents convergence or a provisional scientific score.
        Freshness describes the snapshot's age, not a worker-health diagnosis.
        """
        progress = {
            "availability": "missing", "reason": "state_missing",
            "total": None, "completed": None, "pending": None,
            "attempts": None, "infrastructure_retries": None,
            "updated_at": None, "age_seconds": None, "stale": None,
            "stale_after_seconds": PROGRESS_STALE_SECONDS,
        }
        try:
            self.job_dir(job["evaluation_id"])
            with ExitStack() as opened:
                # Anchor each component to an already opened directory. Refuse
                # symlinks, FIFOs and devices rather than following or blocking
                # on paths substituted while the dashboard is polling.
                directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                directory_fd = os.open(self.root, directory_flags)
                opened.callback(os.close, directory_fd)
                for component in ("jobs", job["evaluation_id"]):
                    directory_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                    opened.callback(os.close, directory_fd)
                state_fd = os.open("evaluation-state.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                   dir_fd=directory_fd)
                stream = opened.enter_context(os.fdopen(state_fd, "rb"))
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    return {**progress, "availability": "unsafe", "reason": "not_regular_file"}
                age = max(0.0, time.time() - before.st_mtime)
                progress.update(updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(before.st_mtime)),
                                age_seconds=round(age, 3), stale=age > PROGRESS_STALE_SECONDS)
                if before.st_size > MAX_PROGRESS_BYTES:
                    return {**progress, "availability": "oversized", "reason": "state_size_limit"}
                raw = stream.read(MAX_PROGRESS_BYTES + 1)
                after = os.fstat(stream.fileno())
                if len(raw) > MAX_PROGRESS_BYTES:
                    return {**progress, "availability": "oversized", "reason": "state_size_limit"}
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    return {**progress, "availability": "invalid", "reason": "state_changed_during_read"}
        except FileNotFoundError:
            return progress
        except RequestError:
            return {**progress, "availability": "unsafe", "reason": "unsafe_state_path"}
        except OSError as exc:
            unsafe = exc.errno in {errno.ELOOP, errno.ENOTDIR}
            return {**progress, "availability": "unsafe" if unsafe else "unreadable",
                    "reason": "unsafe_state_path" if unsafe else "state_read_failed"}
        try:
            state = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            return {**progress, "availability": "invalid", "reason": "invalid_json"}
        try:
            counts = _progress_counts(state, job)
        except ValueError as exc:
            return {**progress, "availability": "invalid", "reason": str(exc)}
        return {**progress, **counts, "availability": "available", "reason": None}

    def request(self, run_id: str, request: dict):
        if not isinstance(request, dict):
            raise RequestError("Request must be a JSON object")
        method = request.get("method")
        fields = {"submit": {"method", "request_id", "source", "split", "candidate_commit"},
                  "status": {"method", "evaluation_id"}, "results": {"method", "evaluation_id"},
                  "resume": {"method", "evaluation_id"}, "cancel": {"method", "evaluation_id"},
                  "diagnostics": {"method", "evaluation_id", "molecule", "artifact_id", "offset", "limit"}}
        if method not in fields or set(request) - fields[method] - {"expected_release_id"}:
            raise RequestError("Unknown method or forbidden request fields")
        if "expected_release_id" in request:
            expected_release = identifier(request["expected_release_id"], "expected_release_id")
            if expected_release != self.config.release_id:
                raise RequestError("Gateway does not serve the approved dataset release")
        if method == "diagnostics":
            # Read stored artifacts without reconciling job state, acquiring its
            # execution lock, touching Redis, or dispatching/resuming work.
            with self.db() as db:
                self._authorize(db, run_id, write=False)
                job = self._owned(db, run_id, request.get("evaluation_id"))
            return self._diagnostics(job, request)
        launch_id = None
        cancel_job = None
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._authorize(db, run_id, write=method in {"submit", "resume"})
            if method == "submit":
                request_id = identifier(request.get("request_id"), "request_id")
                candidate_commit = request.get("candidate_commit", "unknown")
                if candidate_commit != "unknown" and (not isinstance(candidate_commit, str) or
                        not re.fullmatch(r"[0-9a-f]{40,64}", candidate_commit)):
                    raise RequestError("Invalid candidate_commit")
                split = request.get("split", "train")
                if split not in self.config.allowed_splits:
                    raise RequestError("Split is not enabled for this service")
                source = request.get("source")
                if not isinstance(source, str) or len(source.encode("utf-8")) > self.config.max_source_bytes:
                    raise RequestError("Invalid or oversized candidate source")
                try:
                    spec = normalize_optimizer_spec({"kind": "source", "source": source})
                except (ValueError, SyntaxError, RecursionError) as exc:
                    raise RequestError(f"Invalid candidate source: {exc}") from exc
                row = db.execute("SELECT * FROM jobs WHERE run_id=? AND request_id=?", (run_id, request_id)).fetchone()
                if row:
                    job = dict(row)
                    if (job["source_sha256"] != spec["source_sha256"] or job["split"] != split or
                            job["candidate_commit"] != candidate_commit):
                        raise RequestError("request_id already identifies a different candidate or split")
                else:
                    evaluation_id = uuid.uuid4().hex
                    now = time.time()
                    db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                               (evaluation_id, run_id, request_id, spec["source_sha256"], source,
                                split, candidate_commit, "queued", now, now, None, None))
                    job = self._owned(db, run_id, evaluation_id)
                if job["status"] == "queued":
                    launch_id = job["evaluation_id"]
            else:
                job = self._owned(db, run_id, request.get("evaluation_id"))
                if method == "resume" and job["status"] != "complete":
                    # A running process keeps its advisory lock. Duplicate resume
                    # processes exit without touching its durable task identities.
                    if job["status"] == "cancelling":
                        raise RequestError("Cancellation is still settling; retry resume")
                    if job["status"] not in {"running", "queued"}:
                        db.execute("UPDATE jobs SET status='queued', error=NULL, updated=? WHERE evaluation_id=?",
                                   (time.time(), job["evaluation_id"]))
                        job["status"] = "queued"
                    launch_id = job["evaluation_id"]
                elif method == "cancel" and job["status"] != "complete":
                    db.execute("UPDATE jobs SET status='cancelling', updated=? WHERE evaluation_id=?",
                               (time.time(), job["evaluation_id"]))
                    job["status"] = "cancelling"
                    cancel_job = job
        if cancel_job:
            self._finish_cancel(cancel_job)
        if launch_id:
            self.launch(launch_id)
        if method in {"status", "results"} and job["status"] == "queued":
            # A process may die before taking its execution lock. Re-dispatch
            # the same durable job; the lock makes concurrent starts harmless.
            # Keep queued until a child claims it, avoiding a startup grace race.
            self.launch(job["evaluation_id"])
        if method in {"status", "results"} and job["status"] in {"running", "cancelling"}:
            with self.execution_lock(job["evaluation_id"]) as acquired:
                if acquired:
                    if job["status"] == "cancelling":
                        self.cancel_tasks(job, self.job_dir(job["evaluation_id"]) / "evaluation-state.json")
                        with self.db() as db:
                            db.execute("UPDATE jobs SET status='cancelled', updated=? WHERE evaluation_id=? AND status='cancelling'",
                                       (time.time(), job["evaluation_id"]))
                    else:
                        with self.db() as db:
                            db.execute("UPDATE jobs SET status='pending', error='Coordinator interrupted; resume retained task IDs', updated=? WHERE evaluation_id=? AND status='running'",
                                       (time.time(), job["evaluation_id"]))
        with self.db() as db:
            job = self._owned(db, run_id, job["evaluation_id"])
        return self._public(job, include_results=method == "results")

    def _diagnostics(self, job, request):
        from evaluation_access.diagnostics import diagnostic_manifest, read_diagnostic_chunk

        molecule = request.get("molecule")
        if molecule is not None and (
                not isinstance(molecule, str) or not molecule or len(molecule) > 512
                or any(ord(char) < 32 for char in molecule)):
            raise RequestError("Invalid molecule")
        offset = request.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise RequestError("Invalid diagnostic offset")
        artifact_id = request.get("artifact_id")
        maximum = 65536 if artifact_id is not None else 64
        limit = request.get("limit", maximum if artifact_id is not None else 32)
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise RequestError("Invalid diagnostic limit")
        try:
            if artifact_id is None:
                result = diagnostic_manifest(self.config, job, molecule=molecule,
                                             offset=offset, limit=limit)
                result.update(self._diagnostic_capture_status(job, molecule))
            else:
                if molecule is None:
                    raise RequestError("Artifact retrieval requires molecule")
                result = read_diagnostic_chunk(self.config, job, artifact_id,
                                               offset=offset, max_bytes=limit)
                if result.get("identity", {}).get("mol_name") != molecule:
                    raise RequestError("Diagnostic artifact does not belong to molecule")
        except (ValueError, OSError) as exc:
            raise RequestError(str(exc)) from exc
        return {**result, "evaluation_id": job["evaluation_id"],
                "run_id": job["run_id"], "source_sha256": job["source_sha256"],
                "release_id": self.config.release_id,
                "dataset_release_id": self.config.release_id}

    def _diagnostic_capture_status(self, job, molecule):
        """Explain absent uploads from saved receipts; never poll a worker."""
        from distributed_validate.recovery import failure_detail

        path = self.job_dir(job["evaluation_id"]) / "evaluation-state.json"
        output = {"capture_evidence": "not_recorded", "capture_status_counts": {}}
        if not path.exists():
            return output
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Unsafe recovery state")
            with path.open("rb") as stream:
                raw = stream.read(MAX_DIAGNOSTIC_STATE_BYTES + 1)
            if len(raw) > MAX_DIAGNOSTIC_STATE_BYTES:
                raise ValueError("Oversized recovery state")
            state = json.loads(raw)
            if state.get("task_metadata") != {"run_id": job["run_id"],
                                              "evaluation_id": job["evaluation_id"]}:
                raise ValueError("Recovery state ownership mismatch")
            if not isinstance(state.get("records"), list):
                raise ValueError("Malformed recovery state")
            counts = {}
            for record in state["records"]:
                if molecule is not None and record.get("mol_name") != molecule:
                    continue
                detail = failure_detail(record)
                if detail is None:
                    continue
                receipt = detail["diagnostics"]
                status = receipt.get("availability", "not_recorded")
                if not isinstance(status, str) or status not in {
                        "stored", "upload_pending", "unavailable", "disabled", "not_recorded"}:
                    status = "malformed_receipt"
                counts[status] = counts.get(status, 0) + 1
                if molecule is not None:
                    output["capture_receipt"] = {"mol_name": molecule,
                                                  "task_id": detail["task_id"], **receipt}
            output.update(capture_evidence="available", capture_status_counts=counts)
        except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
            output.update(capture_evidence="unavailable", capture_status_counts={})
            output.pop("capture_receipt", None)
        return output

    @contextmanager
    def execution_lock(self, evaluation_id):
        directory = self.job_dir(evaluation_id)
        directory.mkdir(exist_ok=True, mode=0o700)
        with (directory / "execute.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
            else:
                try:
                    yield True
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _finish_cancel(self, job):
        # A final second sweep by the executor closes a cancel/submit race.
        self.cancel_tasks(job, self.job_dir(job["evaluation_id"]) / "evaluation-state.json")
        with self.execution_lock(job["evaluation_id"]) as acquired:
            if acquired:
                with self.db() as db:
                    db.execute("UPDATE jobs SET status='cancelled', updated=? WHERE evaluation_id=? AND status='cancelling'",
                               (time.time(), job["evaluation_id"]))

    def _cancel_tasks(self, job, state_path):
        if not state_path.exists():
            return
        from distributed_validate.client import RemoteOptimizationClient
        client = RemoteOptimizationClient(redis_host=self.config.redis_host,
                                          redis_port=self.config.redis_port,
                                          redis_db=self.config.redis_db)
        state = json.loads(state_path.read_text())
        for record in state.get("records", []):
            for attempt in record.get("attempts", []):
                client.withdraw_queued_task(attempt["task_id"], run_id=job["run_id"],
                                            evaluation_id=job["evaluation_id"])

    def _launch(self, evaluation_id):
        if self.config_path is None:
            raise RuntimeError("A trusted config path is required to launch evaluators")
        directory = self.job_dir(evaluation_id)
        directory.mkdir(exist_ok=True, mode=0o700)
        with (directory / "coordinator.log").open("ab") as log:
            subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "execute",
                              "--config", str(self.config_path.resolve()), "--evaluation-id", evaluation_id],
                             cwd=TRUSTED_REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True, close_fds=True)

    def execute(self, evaluation_id, evaluator_factory=None, record_builder=None):
        with self.execution_lock(evaluation_id) as acquired:
            if not acquired:
                return
            with self.db() as db:
                # Read run permission and move queued -> running in one writer
                # transaction. A concurrent pause/cancel must never be erased
                # by a stale startup snapshot.
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT * FROM jobs WHERE evaluation_id=?", (evaluation_id,)).fetchone()
                if row is None:
                    raise RequestError("Evaluation not found")
                job = dict(row)
                run = db.execute("SELECT state FROM runs WHERE run_id=?", (job["run_id"],)).fetchone()
                if job["status"] in {"complete", "cancelled", "cancelling", "pending"} or run["state"] != "enabled":
                    return
                db.execute("UPDATE jobs SET status='running', updated=? WHERE evaluation_id=?",
                           (time.time(), evaluation_id))

            def should_continue():
                with self.db() as db:
                    current = db.execute("SELECT j.status,r.state FROM jobs j JOIN runs r USING(run_id) WHERE evaluation_id=?",
                                         (evaluation_id,)).fetchone()
                    return current["state"] == "enabled" and current["status"] in {"running", "queued"}

            state_path = self.job_dir(evaluation_id) / "evaluation-state.json"
            started = time.monotonic()
            try:
                if evaluator_factory is None:
                    from validate import Evaluator
                    evaluator_factory = Evaluator
                evaluator = evaluator_factory(molecules_dir=self.config.molecules_dir, split=job["split"],
                                              redis_host=self.config.redis_host, redis_port=self.config.redis_port,
                                              redis_db=self.config.redis_db,
                                              task_timeout_xtb=self.config.task_timeout_seconds)
                result = evaluator.evaluate({"kind": "source", "source": job["source"],
                                             "source_sha256": job["source_sha256"]},
                                            state_path=state_path,
                                            task_metadata={"run_id": job["run_id"], "evaluation_id": evaluation_id},
                                            should_continue=should_continue)
                if record_builder is None:
                    from validate import build_evaluation_record
                    record_builder = build_evaluation_record
                record = record_builder(result, split=job["split"], molecule_count=len(evaluator.molecules),
                                        evaluation_id=evaluation_id, program_id=job["source_sha256"],
                                        candidate_commit=job["candidate_commit"],
                                        release_id=self.config.release_id, dataset_release_id=self.config.release_id,
                                        duration_s=time.monotonic()-started,
                                        evaluation_log=f"{evaluation_id}/run.log")
                status = "complete" if record.get("status") == "complete" else "pending"
                if not should_continue():
                    self.cancel_tasks(job, state_path)
                    status = "cancelled" if status != "complete" else "complete"
                record["status"] = status if status == "complete" else "pending"
                from evaluation_access.client import atomic_json
                atomic_json(self.job_dir(evaluation_id) / "evaluation.json", record)
                if "mean_rel_steps" in record:
                    from validate import _append_run_log
                    _append_run_log(self.job_dir(evaluation_id) / "run.log",
                                    optimizer_description=f"source:{job['source_sha256']}:minimize_func",
                                    split=job["split"], num_molecules=len(evaluator.molecules),
                                    duration_s=record["duration_s"], results=record["results"],
                                    errors=record["errors"], score=record)
                with (self.job_dir(evaluation_id) / "validate_debug.log").open("a", encoding="utf-8") as output:
                    output.write(json.dumps({k: v for k, v in record.items() if k != "results"},
                                            sort_keys=True, allow_nan=False) + "\n")
                with self.db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    current = db.execute("SELECT j.status,r.state FROM jobs j JOIN runs r USING(run_id) WHERE evaluation_id=?",
                                         (evaluation_id,)).fetchone()
                    # Cancellation can arrive after the callback check or log
                    # writes. Never overwrite its accepted transition with a
                    # pending result based on the earlier snapshot.
                    if status != "complete" and (current["status"] == "cancelling" or current["state"] != "enabled"):
                        status = "cancelling"
                    db.execute("UPDATE jobs SET status=?, result=?, error=NULL, updated=? WHERE evaluation_id=?",
                               (status, json.dumps(record, allow_nan=False), time.time(), evaluation_id))
                if status == "cancelling":
                    self.cancel_tasks(job, state_path)
                    with self.db() as db:
                        db.execute("UPDATE jobs SET status='cancelled', updated=? WHERE evaluation_id=? AND status='cancelling'",
                                   (time.time(), evaluation_id))
            except Exception as exc:
                # Infrastructure failures remain resumable; they are never scored
                # as candidate errors. Existing partial results remain durable.
                with self.db() as db:
                    db.execute("UPDATE jobs SET status=CASE WHEN status='cancelling' THEN 'cancelling' ELSE 'pending' END, error=?, updated=? WHERE evaluation_id=?",
                               (f"{type(exc).__name__}: {exc}", time.time(), evaluation_id))
                raise


def read_request(stream):
    raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise RequestError("Request exceeds 2 MiB")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise RequestError("Request must be UTF-8 JSON") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "execute", "provision", "set-state", "status"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--public-key-text")
    parser.add_argument("--state", choices=sorted(RUN_STATES))
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        gateway = Gateway(Config.load(args.config), config_path=args.config)
        if args.action == "serve":
            if os.environ.get("SSH_ORIGINAL_COMMAND", ""):
                raise RequestError("Only the JSON evaluation protocol is allowed")
            output = {"ok": True, "result": gateway.request(args.run_id, read_request(sys.stdin.buffer))}
        elif args.action == "provision":
            line = gateway.provision(args.run_id, args.public_key_text or args.public_key.read_text())
            if gateway.config.authorized_keys:
                state = gateway.run_status(args.run_id)["state"]
                output = {"run_id": args.run_id, "authorized_key_installed": state != "revoked",
                          "state": state}
            else:
                print(line)
                return 0
        elif args.action == "set-state":
            output = gateway.set_run_state(args.run_id, args.state)
        elif args.action == "status":
            output = gateway.run_status(args.run_id)
        else:
            gateway.execute(args.evaluation_id)
            return 0
    except Exception as exc:
        output = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        print(json.dumps(output, allow_nan=False))
        return 1
    print(json.dumps(output, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
