"""Durable, bounded recovery of molecule tasks without partial scientific scores.

Only explicit infrastructure failures and provably lost queued tasks create new
attempt IDs. An expired worker lease is ambiguous and requires diagnosis.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from distributed_validate.client import InfrastructureUnavailable
from distributed_validate.protocol import create_task_id


def _json_default(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot persist {type(value).__name__}")


def evaluation_identity(cases, optimizer_spec, mode="xtb") -> str:
    payload = json.dumps(
        {"cases": cases, "optimizer_spec": optimizer_spec, "mode": mode},
        sort_keys=True, separators=(",", ":"), default=_json_default,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _save(path, state):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, sort_keys=True, default=_json_default)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _transport_error(exc):
    from redis.exceptions import RedisError

    return isinstance(exc, (RedisError, InfrastructureUnavailable, ConnectionError,
                            TimeoutError, OSError))


class _Pending(Exception):
    pass


def _legacy_memory_limit_failure(payload):
    """Recognize the explicit per-task RSS guard from pre-fix workers.

    Unknown child exits and host/transport failures remain infrastructure errors.
    """
    return (isinstance(payload, dict)
            and payload.get("result") is None
            and payload.get("error_kind") == "infrastructure_error"
            and isinstance(payload.get("error"), str)
            and payload["error"].startswith("worker_guard_rss:")
            and isinstance(payload.get("worker_guard"), dict)
            and payload["worker_guard"].get("kind") == "worker_guard_rss")


def _validated_payload(payload, mol_name):
    valid = isinstance(payload, dict)
    if valid:
        result, error = payload.get("result"), payload.get("error")
        kind = payload.get("error_kind")
        success = (isinstance(result, dict) and result.get("mol_name") == mol_name
                   and error is None and kind is None)
        failure = (result is None and isinstance(error, str) and bool(error)
                   and kind in (None, "optimizer_error", "infrastructure_error"))
        valid = success or failure
    if valid:
        if _legacy_memory_limit_failure(payload):
            return {**payload, "error_kind": "optimizer_error"}
        return payload
    return {"result": None, "error_kind": "optimizer_error",
            "error": f"Malformed worker result for molecule {mol_name}"}


def failure_detail(record):
    """Small diagnostic evidence, independent of scientific result accounting.

    Include the latest infrastructure failure while a molecule is pending. Full
    traces are artifact-store objects; never copy them into a result envelope.
    """
    payload = record.get("payload")
    attempts = record.get("attempts", [])
    attempt = next((item for item in reversed(attempts)
                    if item.get("state") == "complete"), None)
    if payload is None:
        attempt = next((item for item in reversed(attempts) if item.get("failure")), None)
        payload = attempt.get("failure") if attempt else None
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not payload.get("error") and not (
            isinstance(result, dict) and result.get("converged") is False):
        return None
    detail = {"mol_name": record["mol_name"],
              "task_id": attempt.get("task_id") if attempt else None,
              "error_kind": payload.get("error_kind"),
              "error": str(payload.get("error") or "")[:1024]}
    if isinstance(result, dict):
        detail.update({key: result.get(key) for key in (
            "stop_reason", "n_steps", "n_completed_steps", "converged")})
    guard = payload.get("worker_guard")
    if isinstance(guard, dict):
        detail["worker_guard"] = {key: value for key, value in guard.items()
                                  if key in {"kind", "exitcode", "elapsed_seconds",
                                             "max_rss_bytes", "max_task_rss_bytes",
                                             "task_timeout_seconds"}
                                  and (value is None or type(value) in (int, float, bool, str))}
        if isinstance(guard.get("traceback"), str):
            detail["traceback"] = guard["traceback"][-4096:]
            detail["traceback_truncated"] = len(guard["traceback"]) > 4096
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        try:
            raw = json.dumps(diagnostics, allow_nan=False)
            detail["diagnostics"] = (diagnostics if len(raw.encode()) <= 3072 else
                                     {"availability": "unavailable", "reason": "oversized_receipt"})
        except (TypeError, ValueError, RecursionError):
            detail["diagnostics"] = {"availability": "unavailable", "reason": "malformed_receipt"}
    else:
        detail["diagnostics"] = {"availability": "not_recorded"}
    return detail


def evaluate_cases(
    client_or_factory, cases, optimizer_spec, state_path, *,
    timeout_seconds=3600.0, mode="xtb", max_attempts=3,
    backoff_seconds=(1.0, 2.0), poll_interval_seconds=0.1,
    task_metadata=None, should_continue=None,
):
    """Return complete results or a resumable pending envelope, never a score.

    A zero-argument client factory allows initial Redis outages to be persisted.
    Network operations have at most ``max_attempts`` attempts per invocation;
    molecule execution attempts are bounded across resumes. Reusing a state path
    with different cases, candidate source, mode, or ownership is rejected.
    """
    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")
    if timeout_seconds < 0 or poll_interval_seconds < 0:
        raise ValueError("Timeout and polling interval must be nonnegative")
    cases = list(cases)
    if set(task_metadata or {}) - {"run_id", "evaluation_id"}:
        raise ValueError("Task metadata may only contain run_id and evaluation_id")
    if len({case["mol_name"] for case in cases}) != len(cases):
        raise ValueError("Molecule names must be unique within an evaluation")
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Evaluation state is already in use: {path}") from exc
        return _evaluate_locked(
            client_or_factory, cases, optimizer_spec, path,
            timeout_seconds=timeout_seconds, mode=mode, max_attempts=max_attempts,
            backoff_seconds=backoff_seconds, poll_interval_seconds=poll_interval_seconds,
            task_metadata=dict(task_metadata or {}), should_continue=should_continue,
        )


def _evaluate_locked(client_or_factory, cases, optimizer_spec, path, *,
                     timeout_seconds, mode, max_attempts, backoff_seconds,
                     poll_interval_seconds, task_metadata, should_continue):
    identity = evaluation_identity(cases, optimizer_spec, mode)
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("version") != 1 or state.get("identity") != identity:
            raise ValueError("Recovery state does not match this evaluation")
        if state.get("task_metadata", {}) != task_metadata:
            raise ValueError("Recovery state ownership does not match this evaluation")
    else:
        state = {
            "version": 1, "identity": identity, "task_metadata": task_metadata,
            "created_at": time.time(), "status": "pending", "events": [],
            "records": [{"mol_name": case["mol_name"], "payload": None, "attempts": [
                {"task_id": create_task_id(mode, case["mol_name"]), "state": "prepared"}
            ]} for case in cases],
        }
        _save(path, state)

    factory = client_or_factory if callable(client_or_factory) else None
    client = None if factory else client_or_factory
    deadline = time.monotonic() + timeout_seconds
    failures = 0

    def event(cause, **details):
        state["events"].append({"timestamp": time.time(), "cause": cause, **details})
        _save(path, state)

    def delay(attempt):
        values = tuple(backoff_seconds)
        if values:
            time.sleep(max(0.0, min(float(values[min(attempt - 1, len(values) - 1)]),
                                   max(0.0, deadline - time.monotonic()))))

    def operation(action, operation_name, task_id=None):
        nonlocal client, failures
        while True:
            try:
                if client is None:
                    client = factory()
                return action(client)
            except Exception as exc:
                if not _transport_error(exc):
                    raise
                failures += 1
                event("transport_failure", operation=operation_name,
                      task_id=task_id, attempt=failures, error=str(exc),
                      will_retry=failures < max_attempts)
                if failures >= max_attempts:
                    raise _Pending("infrastructure_unavailable") from exc
                delay(failures)
                if factory:
                    client = None
                elif hasattr(client, "reconnect"):
                    try:
                        client.reconnect()
                    except Exception as reconnect_error:
                        if not _transport_error(reconnect_error):
                            raise
                        # The next bounded operation performs the actual retry.
                        event("reconnect_failure", error=str(reconnect_error))

    def result_envelope(reason=None):
        complete = all(record["payload"] is not None for record in state["records"])
        state["status"] = "complete" if complete else "pending"
        state["pending_reason"] = None if complete else reason
        state["updated_at"] = time.time()
        results, errors = [], []
        if complete:
            for record in state["records"]:
                payload = _validated_payload(record["payload"], record["mol_name"])
                record["payload"] = payload
                if payload.get("result") is not None:
                    results.append(payload["result"])
                if payload.get("error"):
                    errors.append((record["mol_name"], str(payload["error"])))
        _save(path, state)
        return {
            "status": state["status"], "results": results,
            "num_errors": len(errors), "errors": errors,
            "failure_details": [detail for record in state["records"]
                                if (detail := failure_detail(record)) is not None],
            "recovery_state_path": str(path), "pending_reason": state["pending_reason"],
            "recovery_events": list(state["events"]),
            "infrastructure_retries": sum(max(0, len(record["attempts"]) - 1)
                                          for record in state["records"]),
            "transport_retries": sum(event["cause"] == "transport_failure" and event.get("will_retry", False)
                                     for event in state["events"]),
        }

    try:
        while True:
            if should_continue is not None and not should_continue():
                return result_envelope("paused_or_cancelled")

            # Reconcile every durable ID before dispatching any new work. A
            # coordinator crash after result receipt must never lose that result.
            for record in state["records"]:
                if record["payload"] is not None:
                    continue
                for attempt in record["attempts"]:
                    task_id = attempt["task_id"]
                    payload = operation(lambda c: c.get_result(task_id), "get_result", task_id)
                    # Old pending jobs retain authoritative guard receipts even
                    # after Redis expires them. Do not rerun a known RSS failure.
                    if payload is None and _legacy_memory_limit_failure(attempt.get("failure")):
                        payload = attempt["failure"]
                    if payload is None:
                        continue
                    reclassified_memory_limit = _legacy_memory_limit_failure(payload)
                    payload = _validated_payload(payload, record["mol_name"])
                    if payload.get("error_kind") == "infrastructure_error":
                        if attempt["state"] != "infrastructure_failed":
                            attempt["state"] = "infrastructure_failed"
                            attempt["failure"] = payload
                            event("worker_infrastructure_failure", task_id=task_id,
                                  molecule=record["mol_name"], attempt=record["attempts"].index(attempt) + 1,
                                  error=payload.get("error"))
                        continue
                    record["payload"] = payload
                    attempt["state"] = "complete"
                    if reclassified_memory_limit:
                        event("memory_limit_reclassified_as_optimizer_error",
                              task_id=task_id, molecule=record["mol_name"],
                              previous_error_kind="infrastructure_error")
                    _save(path, state)
                    break

            if all(record["payload"] is not None for record in state["records"]):
                return result_envelope()

            pending_reason = None
            for case, record in zip(cases, state["records"]):
                if record["payload"] is not None:
                    continue
                if should_continue is not None and not should_continue():
                    return result_envelope("paused_or_cancelled")
                attempt = record["attempts"][-1]
                task_id = attempt["task_id"]
                if attempt["state"] not in ("infrastructure_failed", "lost"):
                    observed = operation(lambda c: c.task_status(task_id), "task_status", task_id)
                    observed_state = observed["state"]
                    if observed_state == "lost":
                        attempt["state"] = "lost"
                        event("lost_queued_task", task_id=task_id, evidence=observed)
                    elif observed_state in ("lease_expired", "unknown"):
                        pending_reason = "worker_state_requires_diagnosis"
                        event(pending_reason, task_id=task_id, evidence=observed)
                        continue
                    elif observed_state == "missing" and attempt["state"] != "prepared":
                        pending_reason = "task_state_missing_requires_diagnosis"
                        event(pending_reason, task_id=task_id)
                        continue
                    elif observed_state in ("queued", "running", "finished"):
                        if attempt["state"] != observed_state:
                            attempt["state"] = observed_state
                            _save(path, state)
                        continue
                    elif observed_state == "withdrawn":
                        operation(lambda c: c.resume_queued_task(task_id, **task_metadata),
                                  "resume_queued_task", task_id)
                        attempt["state"] = "submitted"
                        _save(path, state)
                        continue

                if attempt["state"] in ("infrastructure_failed", "lost"):
                    if len(record["attempts"]) >= max_attempts:
                        pending_reason = "worker_attempts_exhausted_requires_diagnosis"
                        continue
                    delay(len(record["attempts"]))
                    attempt = {"task_id": create_task_id(mode, record["mol_name"]),
                               "state": "prepared"}
                    record["attempts"].append(attempt)
                    task_id = attempt["task_id"]
                    _save(path, state)

                def reconcile_and_submit(c):
                    # A lost submit acknowledgement can mean the task already
                    # finished. Always read its existing ID before replaying.
                    if c.get_result(task_id) is not None:
                        return
                    if c.task_status(task_id)["state"] != "missing":
                        return
                    if should_continue is not None and not should_continue():
                        raise _Pending("paused_or_cancelled")
                    # Persist uncertainty before the network call: if its
                    # acknowledgement is lost and Redis later expires the ID,
                    # resume must diagnose the missing task, not submit it again.
                    attempt["state"] = "submitting"
                    _save(path, state)
                    c.submit({**case, **task_metadata, "mode": mode,
                              "optimizer_spec": optimizer_spec}, task_id=task_id)

                operation(reconcile_and_submit, "submit", task_id)
                attempt["state"] = "submitted"
                _save(path, state)

            if pending_reason:
                return result_envelope(pending_reason)
            if time.monotonic() >= deadline:
                return result_envelope("waiting_for_existing_tasks")
            time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
    except _Pending as exc:
        return result_envelope(str(exc))
