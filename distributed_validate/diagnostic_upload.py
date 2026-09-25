"""Optional parent-only diagnostic transfer, isolated from scientific results.

Configure all three GIGAOPT_DIAGNOSTICS_{URL,TOKEN_FILE,SPOOL_DIR} values on the
trusted worker parent. Use HTTPS or an operator-managed localhost SSH tunnel.
No bundle is ever returned to Redis; only an immutable compact receipt is.
"""
from __future__ import annotations

import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from evaluation_access.diagnostics import (
    MAX_BUNDLE_BYTES, _atomic_write, _decode, _directory, _regular_read,
    _run_lock, _validate_identity, _summary, is_truncated, json_bytes, read_token,
)

MAX_SPOOL_BYTES = 64 * 1024 * 1024
MAX_RUN_SPOOL_BYTES = 16 * 1024 * 1024
UPLOAD_TIMEOUT_SECONDS = 2.0
_last_retry = float("-inf")


def capture_enabled() -> bool:
    return all(os.environ.get("GIGAOPT_DIAGNOSTICS_" + key)
               for key in ("URL", "TOKEN_FILE", "SPOOL_DIR"))


def _configuration():
    if not capture_enabled():
        raise ValueError("diagnostics_not_configured")
    endpoint = os.environ["GIGAOPT_DIAGNOSTICS_URL"]
    url = urlsplit(endpoint)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username or url.password or url.query or url.fragment
            or url.path not in {"", "/", "/v1/artifacts"}
            or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("invalid_diagnostics_endpoint")
    endpoint = endpoint.rstrip("/")
    if url.path != "/v1/artifacts":
        endpoint += "/v1/artifacts"
    token_file = Path(os.environ["GIGAOPT_DIAGNOSTICS_TOKEN_FILE"])
    spool = Path(os.environ["GIGAOPT_DIAGNOSTICS_SPOOL_DIR"])
    if not token_file.is_absolute() or not spool.is_absolute() or spool.is_symlink():
        raise ValueError("diagnostic_paths_must_be_absolute")
    # Never resolve or read the token during capture. This function is called
    # only after the untrusted optimization child has exited.
    return endpoint, token_file, spool


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _upload(data, endpoint, token_file):
    token = read_token(token_file)
    request = Request(endpoint, data=data, method="POST", headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json",
        "Connection": "close"})
    try:
        with build_opener(ProxyHandler({}), _NoRedirects()).open(request, timeout=UPLOAD_TIMEOUT_SECONDS) as response:
            raw = response.read(8193)
        if len(raw) > 8192:
            raise ValueError("oversized_collector_receipt")
        value = _decode(raw)
        receipt = value.get("receipt") if isinstance(value, dict) else None
        digest = hashlib.sha256(data).hexdigest()
        identity = _decode(data)["identity"]
        if (not isinstance(value, dict) or not value.get("ok") or not isinstance(receipt, dict)
                or receipt.get("artifact_id") != digest or receipt.get("size_bytes") != len(data)
                or receipt.get("identity") != identity or receipt.get("availability") != "stored"):
            raise ValueError("invalid_collector_receipt")
        return "stored", None
    except HTTPError as exc:
        if exc.code in {400, 401, 403, 404, 413, 507}:
            return "unavailable", "collector_rejected_" + str(exc.code)
        return "upload_pending", "collector_temporarily_unavailable"
    except (URLError, TimeoutError, OSError, HTTPException):
        return "upload_pending", "collector_transport_unavailable"


def _spool_size(root):
    total, runs = 0, {}
    for directory in root.iterdir():
        if directory.is_symlink():
            raise ValueError("invalid_spool_entry")
        if not directory.is_dir():
            continue
        count = 0
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("invalid_spool_entry")
            if path.suffix == ".json":
                count += path.stat().st_size
        runs[directory.name] = count
        total += count
    return total, runs


def _save_spool(spool, run_id, digest, data):
    spool.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory = _directory(spool, run_id, create=True)
    with _run_lock(spool):
        path = directory / (digest + ".json")
        if path.is_symlink():
            raise ValueError("invalid_spool_entry")
        if path.exists():
            if _regular_read(path, MAX_BUNDLE_BYTES) != data:
                raise ValueError("spool_digest_conflict")
            return path
        total, by_run = _spool_size(spool)
        if total + len(data) > MAX_SPOOL_BYTES or by_run.get(run_id, 0) + len(data) > MAX_RUN_SPOOL_BYTES:
            raise ValueError("diagnostic_spool_quota_reached")
        _atomic_write(path, data)
        return path


def retry_pending(*, force=False):
    """Try at most one old bundle per minute; never run inside a calculation."""
    global _last_retry
    now = time.monotonic()
    if not force and now - _last_retry < 60:
        return
    _last_retry = now
    if not capture_enabled():
        return
    try:
        endpoint, token_file, spool = _configuration()
        if not spool.exists():
            return
        _directory(spool)
        for directory in sorted(spool.iterdir()):
            if directory.is_symlink():
                continue
            if not directory.is_dir():
                continue
            _directory(spool, directory.name)
            for path in sorted(directory.glob("*.json")):
                data = _regular_read(path, MAX_BUNDLE_BYTES)
                if path.stem != hashlib.sha256(data).hexdigest():
                    continue
                availability, _reason = _upload(data, endpoint, token_file)
                if availability != "upload_pending":
                    path.unlink(missing_ok=True)
                return
    except Exception:
        # A missing artifact never changes an optimization result or retries it.
        return


def publish(task, payload):
    """Strip the private bundle unconditionally; add only a bounded receipt."""
    bundle = payload.pop("_diagnostic_bundle", None)
    guard = payload.get("worker_guard")
    if isinstance(guard, dict) and isinstance(guard.get("traceback"), str):
        encoded = guard["traceback"].encode("utf-8", errors="replace")
        guard["traceback"] = encoded[-8192:].decode("utf-8", errors="ignore")
        if len(encoded) > 8192:
            guard["traceback_truncated"] = True
    if bundle is None:
        result = payload.get("result")
        if payload.get("error") or (isinstance(result, dict) and not result.get("converged", False)):
            enabled = capture_enabled()
            payload.setdefault("diagnostics", {"schema_version": 1,
                "availability": "unavailable" if enabled else "disabled",
                "reason": "capture_missing" if enabled else "diagnostics_not_configured",
                "summary": _summary(result) if isinstance(result, dict) else {}, "truncated": False})
        return payload
    descriptor = {"schema_version": 1, "availability": "unavailable", "truncated": False}
    payload["diagnostics"] = descriptor
    try:
        source = task.get("optimizer_spec", {}).get("source_sha256")
        if source is None and isinstance(task.get("optimizer_spec", {}).get("source"), str):
            source = hashlib.sha256(task["optimizer_spec"]["source"].encode()).hexdigest()
        identity = {"run_id": task.get("run_id"), "evaluation_id": task.get("evaluation_id"),
                    "task_id": task.get("task_id"), "source_sha256": source,
                    "mol_name": task.get("mol_name")}
        _validate_identity(identity)
        if not isinstance(bundle, dict):
            raise ValueError("invalid_capture_bundle")
        descriptor["summary"] = _summary(bundle)
        data = json_bytes({"schema_version": 1, "identity": identity, "trace": bundle})
        if len(data) > MAX_BUNDLE_BYTES:
            raise ValueError("diagnostic_bundle_too_large")
        digest = hashlib.sha256(data).hexdigest()
        descriptor.update(identity=identity, artifact_id=digest, size_bytes=len(data),
                          truncated=is_truncated(bundle))
        if not capture_enabled():
            descriptor.update(availability="disabled", reason="diagnostics_not_configured")
            return payload
        endpoint, token_file, spool = _configuration()
        path = _save_spool(spool, identity["run_id"], digest, data)
        availability, reason = _upload(data, endpoint, token_file)
        descriptor["availability"] = availability
        if reason:
            descriptor["reason"] = reason
        if availability != "upload_pending":
            path.unlink(missing_ok=True)
    except ValueError as exc:
        allowed = {"diagnostic_spool_quota_reached", "diagnostic_bundle_too_large",
                   "invalid_capture_bundle", "invalid_collector_receipt", "oversized_collector_receipt",
                   "invalid_diagnostics_endpoint", "diagnostic_paths_must_be_absolute"}
        reason = str(exc) if str(exc) in allowed else "diagnostic_capture_or_transfer_unavailable"
        descriptor.update(availability="unavailable", reason=reason)
    except Exception:
        descriptor.update(availability="unavailable", reason="diagnostic_capture_or_transfer_unavailable")
    return payload
