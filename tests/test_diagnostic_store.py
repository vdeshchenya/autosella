"""Offline artifact-store and transfer tests; no Redis or numerical work."""
import base64
from dataclasses import replace
import hashlib
from http.client import IncompleteRead
import json
import os
import time
from pathlib import Path
import threading
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from evaluation_access.diagnostics import (
    ArtifactStore, DiagnosticQuotaError, MAX_BUNDLE_BYTES, MAX_RESPONSE_BYTES,
    collector_server, diagnostic_manifest, json_bytes, read_diagnostic_chunk,
)
from evaluation_access.server import Config, Gateway, RequestError
from distributed_validate import diagnostic_upload


@pytest.fixture
def prepared(tmp_path):
    cfg = Config(state_root=tmp_path / "state", molecules_dir=tmp_path / "molecules",
                 release_id="fixture", diagnostics_max_run_bytes=1024 * 1024,
                 diagnostics_token_file=tmp_path / "token")
    cfg.diagnostics_token_file.write_text("a" * 64)
    gateway = Gateway(cfg, launch=lambda _evaluation: None)
    with gateway.db() as db:
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("run-a", "enabled", "key", 1, None))
    job = gateway.request("run-a", {"method": "submit", "source": "def minimize_func(*a, **k): pass\n",
                                   "request_id": "first", "candidate_commit": "1" * 40})
    directory = gateway.job_dir(job["evaluation_id"])
    directory.mkdir(exist_ok=True)
    task_id = "xtb:molecule:01234567"
    state = {"version": 1, "task_metadata": {"run_id": "run-a", "evaluation_id": job["evaluation_id"]},
             "records": [{"mol_name": "molecule", "payload": None,
                          "attempts": [{"task_id": task_id, "state": "submitted"}]}]}
    (directory / "evaluation-state.json").write_bytes(json_bytes(state))
    identity = {"run_id": "run-a", "evaluation_id": job["evaluation_id"],
                "task_id": task_id, "source_sha256": job["source_sha256"], "mol_name": "molecule"}
    bundle = {"schema_version": 1, "identity": identity,
              "trace": {"attempted_calls": 4, "completed_calls": 3,
                        "last_attempt": {"call_index": 4, "positions": [[1, 2, 3]]},
                        "truncation": {"older_completed_frames_not_retained": 0}}}
    return cfg, job, bundle


def test_store_roundtrip_immutable_and_read_only(prepared):
    cfg, job, bundle = prepared
    before = list(cfg.state_root.iterdir())
    assert diagnostic_manifest(cfg, job)["artifacts"] == []
    assert list(cfg.state_root.iterdir()) == before
    data = json_bytes(bundle)
    first = ArtifactStore(cfg).ingest(data)
    assert ArtifactStore(cfg).ingest(data) == first
    manifest = diagnostic_manifest(cfg, job, molecule="molecule")
    assert manifest["total"] == 1 and manifest["eof"]
    assert manifest["artifacts"][0]["summary"]["last_attempted_call_index"] == 4
    part = read_diagnostic_chunk(cfg, job, first["artifact_id"], max_bytes=11)
    restored = base64.b64decode(part["data"])
    while not part["eof"]:
        part = read_diagnostic_chunk(cfg, job, first["artifact_id"], offset=part["next_offset"], max_bytes=11)
        restored += base64.b64decode(part["data"])
    assert restored == data
    assert part["identity"] == bundle["identity"]


@pytest.mark.parametrize("key,value", [("run_id", "other"), ("source_sha256", "0" * 64),
                                       ("task_id", "xtb:unknown"), ("mol_name", "other")])
def test_upload_identity_requires_durable_owned_task(prepared, key, value):
    cfg, _, bundle = prepared
    bundle["identity"][key] = value
    with pytest.raises(ValueError):
        ArtifactStore(cfg).ingest(json_bytes(bundle))
    assert not (cfg.state_root / "diagnostics").exists()


def test_store_caps_idempotence_and_paging(prepared):
    cfg, job, bundle = prepared
    store = ArtifactStore(cfg)
    bundle["trace"]["text"] = "a" * 600_000
    first = store.ingest(json_bytes(bundle))
    bundle["trace"]["text"] = "b" * 600_000
    second = store.ingest(json_bytes(bundle))
    assert diagnostic_manifest(cfg, job)["artifacts"][0]["artifact_id"] == second["artifact_id"]
    with pytest.raises(ValueError, match="unavailable"):
        read_diagnostic_chunk(cfg, job, first["artifact_id"])
    bundle["trace"]["text"] = "a" * 600_000
    assert store.ingest(json_bytes(bundle))["artifact_id"] == first["artifact_id"]
    with pytest.raises(ValueError, match="1 MiB"):
        store.ingest(b" " * (MAX_BUNDLE_BYTES + 1))
    part = read_diagnostic_chunk(cfg, job, first["artifact_id"])
    assert len(json_bytes(part)) <= MAX_RESPONSE_BYTES and not part["eof"]
    with pytest.raises(ValueError):
        diagnostic_manifest(cfg, job, limit=65)
    with pytest.raises(ValueError):
        read_diagnostic_chunk(cfg, job, "../elsewhere")


def test_manifest_has_bounded_pages(prepared):
    cfg, job, bundle = prepared
    for number in range(35):
        bundle["trace"]["attempted_calls"] = number
        ArtifactStore(cfg).ingest(json_bytes(bundle))
    first = diagnostic_manifest(cfg, job, limit=32)
    assert len(first["artifacts"]) == 32 and first["next_offset"] == 32 and not first["eof"]
    last = diagnostic_manifest(cfg, job, offset=32, limit=32)
    assert len(last["artifacts"]) == 3 and last["eof"] and last["total"] == 35
    assert len(json_bytes(first)) <= MAX_RESPONSE_BYTES


def test_symlink_and_integrity_rejected(prepared, tmp_path):
    cfg, job, bundle = prepared
    receipt = ArtifactStore(cfg).ingest(json_bytes(bundle))
    directory = cfg.state_root / "diagnostics" / "run-a" / job["evaluation_id"]
    artifact = directory / (receipt["artifact_id"] + ".json")
    artifact.write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        read_diagnostic_chunk(cfg, job, receipt["artifact_id"])
    artifact.unlink()
    artifact.symlink_to(tmp_path / "outside")
    with pytest.raises((ValueError, OSError)):
        read_diagnostic_chunk(cfg, job, receipt["artifact_id"])
    with pytest.raises(ValueError, match="symlink"):
        ArtifactStore(cfg).ingest(json_bytes(bundle))
    artifact.unlink()
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    directory.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        diagnostic_manifest(cfg, job)


def test_collector_auth_and_upload(prepared):
    cfg, job, bundle = prepared
    with collector_server(cfg) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/v1/artifacts"
        try:
            bad = Request(url, data=json_bytes(bundle), headers={"Content-Type": "application/json"})
            with pytest.raises(HTTPError) as error:
                urlopen(bad, timeout=2)
            assert error.value.code == 401
            state, reason = diagnostic_upload._upload(json_bytes(bundle), url, cfg.diagnostics_token_file)
            assert (state, reason) == ("stored", None)
            assert diagnostic_manifest(cfg, job)["total"] == 1
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _task(bundle):
    identity = bundle["identity"]
    return {**identity, "optimizer_spec": {"source_sha256": identity["source_sha256"]}}


def _upload_environment(monkeypatch, cfg, tmp_path):
    monkeypatch.setenv("GIGAOPT_DIAGNOSTICS_URL", "http://127.0.0.1:12345")
    monkeypatch.setenv("GIGAOPT_DIAGNOSTICS_TOKEN_FILE", str(cfg.diagnostics_token_file))
    monkeypatch.setenv("GIGAOPT_DIAGNOSTICS_SPOOL_DIR", str(tmp_path / "spool"))


def test_uploader_never_returns_bundle_and_disabled_receipt(prepared, monkeypatch):
    _, _, bundle = prepared
    for name in ("URL", "TOKEN_FILE", "SPOOL_DIR"):
        monkeypatch.delenv("GIGAOPT_DIAGNOSTICS_" + name, raising=False)
    result = {"result": None, "error": "optimizer_error", "error_kind": "optimizer_error",
              "_diagnostic_bundle": bundle["trace"]}
    assert diagnostic_upload.publish(_task(bundle), result) is result
    assert "_diagnostic_bundle" not in result
    assert result["diagnostics"]["availability"] == "disabled"
    assert result["diagnostics"]["summary"]["attempted_calls"] == 4
    assert result["error"] == "optimizer_error" and result["result"] is None
    uncaptured = diagnostic_upload.publish({}, {"result": None, "error": "failure"})
    assert uncaptured["diagnostics"]["availability"] == "disabled"


def test_upload_pending_retry_and_quota_do_not_change_result(prepared, tmp_path, monkeypatch):
    cfg, _, bundle = prepared
    _upload_environment(monkeypatch, cfg, tmp_path)
    monkeypatch.setattr(diagnostic_upload, "_upload", lambda *_a: ("upload_pending", "offline"))
    payload = {"result": {"converged": False, "n_steps": 4}, "error": None,
               "_diagnostic_bundle": bundle["trace"]}
    diagnostic_upload.publish(_task(bundle), payload)
    assert payload["diagnostics"]["availability"] == "upload_pending"
    assert len(list((tmp_path / "spool").glob("*/*.json"))) == 1
    monkeypatch.setattr(diagnostic_upload, "_upload", lambda *_a: ("stored", None))
    diagnostic_upload.retry_pending(force=True)
    assert list((tmp_path / "spool").glob("*/*.json")) == []
    monkeypatch.setattr(diagnostic_upload, "MAX_RUN_SPOOL_BYTES", 1)
    new = {"result": None, "error": "same", "_diagnostic_bundle": bundle["trace"]}
    diagnostic_upload.publish(_task(bundle), new)
    assert new["error"] == "same" and new["diagnostics"]["availability"] == "unavailable"
    assert "_diagnostic_bundle" not in new


def test_unexpected_upload_failure_cannot_break_scientific_result(prepared, tmp_path, monkeypatch):
    cfg, _, bundle = prepared
    _upload_environment(monkeypatch, cfg, tmp_path)
    def broken(*_a):
        raise RuntimeError("secret/path must not escape")
    monkeypatch.setattr(diagnostic_upload, "_upload", broken)
    payload = {"result": None, "error": "scientific", "_diagnostic_bundle": bundle["trace"]}
    diagnostic_upload.publish(_task(bundle), payload)
    assert payload["error"] == "scientific"
    assert "secret/path" not in json.dumps(payload)
    assert "_diagnostic_bundle" not in payload


def test_insecure_remote_http_configuration_is_rejected(prepared, tmp_path, monkeypatch):
    cfg, _, bundle = prepared
    _upload_environment(monkeypatch, cfg, tmp_path)
    monkeypatch.setenv("GIGAOPT_DIAGNOSTICS_URL", "http://public.example/artifacts")
    with pytest.raises(ValueError, match="endpoint"):
        diagnostic_upload._configuration()


@pytest.mark.parametrize("value", [b"[]", b"null", b"{broken", IncompleteRead(b"partial", 20)])
def test_malformed_collector_response_never_changes_result(prepared, tmp_path, monkeypatch, value):
    cfg, _, bundle = prepared
    _upload_environment(monkeypatch, cfg, tmp_path)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_a): pass
        def read(self, _limit):
            if isinstance(value, Exception):
                raise value
            return value
    class Opener:
        def open(self, *_a, **_kw): return Response()
    monkeypatch.setattr(diagnostic_upload, "build_opener", lambda *_a: Opener())
    payload = {"result": None, "error": "original", "error_kind": "optimizer_error",
               "_diagnostic_bundle": bundle["trace"]}
    diagnostic_upload.publish(_task(bundle), payload)
    assert payload["error"] == "original" and payload["result"] is None
    assert "_diagnostic_bundle" not in payload
    assert payload["diagnostics"]["availability"] in {"unavailable", "upload_pending"}
    diagnostic_upload.retry_pending(force=True)


def test_existing_traceback_is_bounded_before_redis(prepared, monkeypatch):
    _, _, bundle = prepared
    monkeypatch.delenv("GIGAOPT_DIAGNOSTICS_URL", raising=False)
    payload = {"result": None, "error": "failure", "worker_guard": {"traceback": "x" * 20000}}
    diagnostic_upload.publish(_task(bundle), payload)
    assert len(payload["worker_guard"]["traceback"].encode()) <= 8192
    assert payload["worker_guard"]["traceback_truncated"]


def test_collector_fixed_path_isolated_launch_imports_config(prepared, tmp_path):
    cfg, _, _ = prepared
    script = Path(__file__).resolve().parents[1] / "evaluation_access/diagnostics.py"
    help_run = subprocess.run([sys.executable, "-I", str(script), "--help"],
                              capture_output=True, text=True, timeout=10)
    assert help_run.returncode == 0 and "--config" in help_run.stdout
    broken = tmp_path / "bad-config.json"
    broken.write_text('{}')
    run = subprocess.run([sys.executable, "-I", str(script), "--config", str(broken), "--port", "0"],
                         capture_output=True, text=True, timeout=10)
    assert run.returncode != 0 and "ModuleNotFoundError" not in run.stderr
    assert "state_root" in run.stderr


def test_collector_storage_outage_keeps_worker_spool(prepared, tmp_path, monkeypatch):
    cfg, _, bundle = prepared
    def unavailable(*_a):
        raise OSError("temporary disk outage")
    monkeypatch.setattr(ArtifactStore, "ingest", unavailable)
    with collector_server(cfg) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _upload_environment(monkeypatch, cfg, tmp_path)
        monkeypatch.setenv("GIGAOPT_DIAGNOSTICS_URL", f"http://127.0.0.1:{server.server_port}")
        try:
            payload = {"result": None, "error": "original", "_diagnostic_bundle": bundle["trace"]}
            diagnostic_upload.publish(_task(bundle), payload)
            assert payload["diagnostics"]["availability"] == "upload_pending"
            assert len(list((tmp_path / "spool").glob("*/*.json"))) == 1
            assert payload["error"] == "original"
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_expiry_does_not_renew_on_download_or_duplicate(prepared, monkeypatch):
    cfg, job, bundle = prepared
    now = [1_800_000_000.0]
    monkeypatch.setattr("evaluation_access.diagnostics.time.time", lambda: now[0])
    store = ArtifactStore(cfg)
    first = store.ingest(json_bytes(bundle))
    directory = cfg.state_root / "diagnostics" / "run-a" / job["evaluation_id"]
    now[0] += cfg.diagnostics_retention_seconds - 1
    assert store.ingest(json_bytes(bundle)) == first
    read_diagnostic_chunk(cfg, job, first["artifact_id"])
    store.cleanup()
    assert diagnostic_manifest(cfg, job)["total"] == 1
    now[0] += 1
    store.cleanup()
    assert not directory.exists()  # Both payload and metadata, no tombstone.
    assert diagnostic_manifest(cfg, job, offset=32)["eof"]
    with pytest.raises(ValueError, match="expired, evicted"):
        read_diagnostic_chunk(cfg, job, first["artifact_id"])
    gateway = Gateway(cfg, launch=lambda _: None)
    with pytest.raises(RequestError, match="unavailable"):
        gateway.request("run-a", {"method": "diagnostics", "evaluation_id": job["evaluation_id"],
                                  "molecule": "molecule", "artifact_id": first["artifact_id"]})
    # Historical receipts remain valid records of upload, not promises of retention.
    assert first["availability"] == "stored"


def test_idle_collector_expires_legacy_payloads(prepared, monkeypatch):
    cfg, job, bundle = prepared
    receipt = ArtifactStore(cfg).ingest(json_bytes(bundle))
    directory = cfg.state_root / "diagnostics" / "run-a" / job["evaluation_id"]
    meta = directory / (receipt["artifact_id"] + ".meta")
    metadata = json.loads(meta.read_bytes())
    metadata.pop("ingested_at")
    meta.write_bytes(json_bytes(metadata))
    payload = meta.with_suffix(".json")
    old = time.time() - cfg.diagnostics_retention_seconds - 1
    os.utime(payload, (old, old))
    # Metadata was just written: legacy expiry must use payload mtime.
    monkeypatch.setattr("evaluation_access.diagnostics.MAINTENANCE_INTERVAL_SECONDS", 0.01)
    with collector_server(cfg) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 2
            while directory.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not directory.exists()
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_quota_evicts_oldest_across_evaluations_without_renewing_reads(prepared, monkeypatch):
    cfg, job, bundle = prepared
    now = [time.time()]
    monkeypatch.setattr("evaluation_access.diagnostics.time.time", lambda: now[0])
    store = ArtifactStore(cfg)
    gateway = Gateway(cfg, launch=lambda _: None)
    def another_job(run_id, request_id):
        other = gateway.request(run_id, {"method": "submit", "source": "def minimize_func(*a, **k): pass\n",
                                          "request_id": request_id, "candidate_commit": "1" * 40})
        directory = gateway.job_dir(other["evaluation_id"])
        directory.mkdir(exist_ok=True)
        original = gateway.job_dir(job["evaluation_id"]) / "evaluation-state.json"
        state = json.loads(original.read_bytes())
        state["task_metadata"] = {"run_id": run_id, "evaluation_id": other["evaluation_id"]}
        (directory / "evaluation-state.json").write_bytes(json_bytes(state))
        return other
    second_job = another_job("run-a", "second")
    with gateway.db() as db:
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("run-b", "enabled", "key-b", 1, None))
    separate_job = another_job("run-b", "separate")
    separate_bundle = json.loads(json_bytes(bundle))
    separate_bundle["identity"].update(run_id="run-b", evaluation_id=separate_job["evaluation_id"])
    separate_receipt = store.ingest(json_bytes(separate_bundle))
    receipts = []
    for index in range(3):
        now[0] += 1
        bundle["identity"]["evaluation_id"] = (job if index == 0 else second_job)["evaluation_id"]
        bundle["trace"]["text"] = str(index) * 400_000
        receipts.append(store.ingest(json_bytes(bundle)))
        if index == 1:
            read_diagnostic_chunk(cfg, job, receipts[0]["artifact_id"])
    assert diagnostic_manifest(cfg, job)["total"] == 0
    stored = {m["artifact_id"] for m in diagnostic_manifest(cfg, second_job)["artifacts"]}
    assert stored == {r["artifact_id"] for r in receipts[1:]}
    assert diagnostic_manifest(cfg, separate_job)["artifacts"][0]["artifact_id"] == separate_receipt["artifact_id"]
    directory = cfg.state_root / "diagnostics" / "run-a"
    assert sum(p.stat().st_size for p in directory.glob("*/*")) <= cfg.diagnostics_max_run_bytes
    assert not list(directory.glob("*/" + receipts[0]["artifact_id"] + ".*"))
    # An impossible new artifact does not destroy the retained ones.
    tiny = ArtifactStore(replace(cfg, diagnostics_max_run_bytes=1))
    with pytest.raises(DiagnosticQuotaError, match="cannot fit"):
        tiny.ingest(json_bytes(bundle))
    assert {m["artifact_id"] for m in diagnostic_manifest(cfg, second_job)["artifacts"]} == stored


def test_cleanup_waits_for_chunk_read_and_deletes_pairs(prepared, monkeypatch):
    from evaluation_access import diagnostics
    cfg, job, bundle = prepared
    receipt = ArtifactStore(cfg).ingest(json_bytes(bundle))
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    original_read = diagnostics._regular_read
    def held_read(path, limit):
        if path.name == receipt["artifact_id"] + ".json":
            entered.set()
            assert release.wait(2)
        return original_read(path, limit)
    monkeypatch.setattr(diagnostics, "_regular_read", held_read)
    chunks, failures = [], []
    def reader():
        try:
            chunks.append(read_diagnostic_chunk(cfg, job, receipt["artifact_id"]))
        except Exception as exc:
            failures.append(exc)
    def cleanup():
        try:
            ArtifactStore(replace(cfg, diagnostics_retention_seconds=0)).cleanup()
        except Exception as exc:
            failures.append(exc)
        finally:
            cleaned.set()
    reader_thread = threading.Thread(target=reader)
    cleaner_thread = threading.Thread(target=cleanup)
    reader_thread.start()
    assert entered.wait(2)
    cleaner_thread.start()
    try:
        assert not cleaned.wait(0.05)
    finally:
        release.set()
        reader_thread.join(timeout=2)
        cleaner_thread.join(timeout=2)
    assert not failures and chunks and cleaned.is_set()
    assert diagnostic_manifest(cfg, job)["total"] == 0


def test_sweep_repairs_half_deleted_pair_and_refuses_symlink(prepared, tmp_path):
    cfg, job, bundle = prepared
    receipt = ArtifactStore(cfg).ingest(json_bytes(bundle))
    directory = cfg.state_root / "diagnostics" / "run-a" / job["evaluation_id"]
    payload = directory / (receipt["artifact_id"] + ".json")
    payload.unlink()
    assert diagnostic_manifest(cfg, job)["total"] == 0
    ArtifactStore(cfg).cleanup()
    assert not directory.exists()
    ArtifactStore(cfg).ingest(json_bytes(bundle))
    outside = tmp_path / "must-stay"
    outside.write_text("preserve")
    payload.unlink()
    payload.symlink_to(outside)
    ArtifactStore(replace(cfg, diagnostics_retention_seconds=0)).cleanup()
    assert outside.read_text() == "preserve" and payload.is_symlink()
    assert payload.with_suffix(".meta").exists()  # Run rejected before any deletion.
