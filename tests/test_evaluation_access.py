"""Offline tests of the SSH authority boundary and reconnectable gateway."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from distributed_validate.optimizer import describe_optimizer_spec, normalize_optimizer_spec, optimizer_cache_key
from evaluation_access.client import Client, evaluate, ssh_command, TransportError
from evaluation_access.server import Config, Gateway, MAX_REQUEST_BYTES, RequestError, read_request


SOURCE = "def minimize_func(system):\n    return system\n"


def public_key(byte=1):
    return "ssh-ed25519 " + base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes([byte]) * 32).decode()


@pytest.fixture
def gateway(tmp_path):
    config = Config(state_root=tmp_path / "state", molecules_dir=tmp_path / "molecules",
                    release_id="fixture-release", authorized_keys=tmp_path / "ssh" / "authorized_keys")
    launched, cancelled = [], []
    value = Gateway(config, config_path=tmp_path / "config.json", launch=launched.append,
                    cancel_tasks=lambda job, path: cancelled.append((job["run_id"], job["evaluation_id"])))
    value.provision("run-a", public_key(1))
    value.provision("run-b", public_key(2))
    value.launched, value.cancelled = launched, cancelled
    return value


def submit(gateway, owner="run-a", request_id="trial-1", **extra):
    return gateway.request(owner, {"method": "submit", "request_id": request_id,
                                   "source": SOURCE, "split": "train", **extra})


def test_idempotent_submit_and_source_not_executed(gateway, tmp_path):
    marker = tmp_path / "source-was-executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n" + SOURCE
    one = submit(gateway, source=source)
    two = submit(gateway, source=source)
    assert one["evaluation_id"] == two["evaluation_id"]
    assert one["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert not marker.exists()
    with pytest.raises(RequestError, match="different candidate"):
        submit(gateway, source=SOURCE)
    with gateway.db() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


@pytest.mark.parametrize("method", ["status", "results", "cancel", "resume", "diagnostics"])
def test_every_lookup_is_owned(gateway, method):
    job = submit(gateway)
    with pytest.raises(RequestError, match="not found"):
        gateway.request("run-b", {"method": method, "evaluation_id": job["evaluation_id"]})
    assert not gateway.cancelled


def test_diagnostics_never_reconciles_or_launches_queued_job(gateway):
    job = submit(gateway)
    with gateway.db() as db:
        before = dict(db.execute("SELECT * FROM jobs WHERE evaluation_id=?",
                                 (job["evaluation_id"],)).fetchone())
    gateway.launched.clear()
    receipt = gateway.request("run-a", {"method": "diagnostics", "evaluation_id": job["evaluation_id"]})
    assert receipt["artifacts"] == []
    assert receipt["capture_evidence"] == "not_recorded"
    assert receipt["release_id"] == receipt["dataset_release_id"] == "fixture-release"
    assert gateway.launched == gateway.cancelled == []
    with gateway.db() as db:
        after = dict(db.execute("SELECT * FROM jobs WHERE evaluation_id=?",
                                (job["evaluation_id"],)).fetchone())
    assert before == after
    assert not gateway.job_dir(job["evaluation_id"]).exists()


def test_diagnostic_bundle_is_owned_bounded_and_readable_while_paused(gateway):
    from evaluation_access.diagnostics import ArtifactStore, json_bytes
    from evaluation_access.client import diagnostics

    job = submit(gateway)
    identity = {"run_id": "run-a", "evaluation_id": job["evaluation_id"],
                "source_sha256": job["source_sha256"], "mol_name": "mol-a", "task_id": "task-a"}
    directory = gateway.job_dir(job["evaluation_id"])
    directory.mkdir()
    state = {"version": 1, "task_metadata": {"run_id": "run-a", "evaluation_id": job["evaluation_id"]},
             "records": [{"mol_name": "mol-a", "attempts": [{"task_id": "task-a", "state": "complete"}],
                          "payload": {"result": None, "error": "SCF failed", "error_kind": "optimizer_error",
                                      "diagnostics": {"availability": "upload_pending"}}}]}
    (directory / "evaluation-state.json").write_text(json.dumps(state))
    bundle = json_bytes({"schema_version": 1, "identity": identity,
                         "trace": {"n_steps": 2, "n_completed_steps": 1, "truncated": False,
                                   "text": "α" * 50000}})
    stored = ArtifactStore(gateway.config).ingest(bundle)
    gateway.set_run_state("run-a", "paused")
    gateway.launched.clear()
    request = {"method": "diagnostics", "evaluation_id": job["evaluation_id"], "molecule": "mol-a"}
    manifest = gateway.request("run-a", request)
    assert manifest["artifacts"][0]["artifact_id"] == stored["artifact_id"]
    assert manifest["capture_status_counts"] == {"upload_pending": 1}
    assert manifest["capture_receipt"]["task_id"] == "task-a"
    offset, chunks = 0, []
    while True:
        page = gateway.request("run-a", {**request, "artifact_id": stored["artifact_id"], "offset": offset})
        assert len(json.dumps(page).encode()) <= 65536
        assert page["identity"] == identity and page["offset"] == offset
        chunks.append(base64.b64decode(page["data"]))
        if page["eof"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert b"".join(chunks) == bundle
    assert not gateway.launched
    class LocalClient:
        expected_release_id = "fixture-release"

        def request(self, body):
            assert body["method"] == "diagnostics"
            return gateway.request("run-a", body)

    destination = gateway.root.parent / "downloaded"
    manifest_download = diagnostics(LocalClient(), evaluation_id=job["evaluation_id"],
                                    molecule="mol-a", output_dir=destination)
    assert manifest_download["total_artifacts"] == 1
    downloaded = diagnostics(LocalClient(), evaluation_id=job["evaluation_id"], molecule="mol-a",
                             artifact_id=stored["artifact_id"], output_dir=destination)
    assert Path(downloaded["diagnostic_artifact"]).read_bytes() == bundle
    assert not gateway.launched
    with pytest.raises(RequestError, match="molecule"):
        gateway.request("run-a", {**request, "molecule": "other", "artifact_id": stored["artifact_id"]})
    with pytest.raises(RequestError, match="not found"):
        gateway.request("run-b", {**request, "artifact_id": stored["artifact_id"]})


@pytest.mark.parametrize("extra", [{"offset": -1}, {"offset": True}, {"limit": 65},
                                  {"limit": False}, {"molecule": "bad\nname"},
                                  {"artifact_id": "../escape", "molecule": "a"},
                                  {"artifact_id": "a" * 64}])
def test_diagnostic_request_rejects_invalid_selection_without_dispatch(gateway, extra):
    job = submit(gateway)
    gateway.launched.clear()
    with pytest.raises(RequestError):
        gateway.request("run-a", {"method": "diagnostics", "evaluation_id": job["evaluation_id"], **extra})
    assert not gateway.launched


@pytest.mark.parametrize("field,value", [("run_id", "run-b"), ("redis_host", "elsewhere"),
                                        ("molecules_dir", "/tmp"), ("evaluator", "evil.py"),
                                        ("program_path", "../../algo.py"), ("command", "id"),
                                        ("module_name", "os")])
def test_untrusted_authority_and_path_fields_rejected(gateway, field, value):
    with pytest.raises(RequestError, match="forbidden"):
        submit(gateway, **{field: value})


@pytest.mark.parametrize("request_id", ["../evil", "/tmp/a", "", "x\ny", "x" * 97])
def test_request_ids_are_not_paths(gateway, request_id):
    with pytest.raises(RequestError, match="request_id"):
        submit(gateway, request_id=request_id)


def test_invalid_source_split_and_bounded_json(gateway):
    with pytest.raises(RequestError, match="candidate source"):
        submit(gateway, source="invalid Python !")
    with pytest.raises(RequestError, match="oversized"):
        submit(gateway, source=" " * (1024 * 1024 + 1))
    with pytest.raises(RequestError, match="Split"):
        submit(gateway, split="test")
    with pytest.raises(RequestError, match="2 MiB"):
        read_request(io.BytesIO(b" " * (MAX_REQUEST_BYTES + 1)))
    with pytest.raises(RequestError, match="JSON"):
        read_request(io.BytesIO(b"\xff"))
    with pytest.raises(RequestError, match="evaluation_id"):
        gateway.request("run-a", {"method": "results", "evaluation_id": "../gateway.sqlite3"})


def test_cancel_resume_and_admin_pause_are_scoped(gateway):
    first = submit(gateway)
    other = submit(gateway, owner="run-b")
    gateway.request("run-a", {"method": "cancel", "evaluation_id": first["evaluation_id"]})
    assert gateway.cancelled == [("run-a", first["evaluation_id"])]
    assert gateway.request("run-b", {"method": "status", "evaluation_id": other["evaluation_id"]})["status"] == "queued"
    gateway.request("run-a", {"method": "resume", "evaluation_id": first["evaluation_id"]})
    assert gateway.launched[-1] == first["evaluation_id"]
    gateway.set_run_state("run-a", "paused")
    with pytest.raises(RequestError, match="paused"):
        submit(gateway, request_id="new")
    with pytest.raises(RequestError, match="paused"):
        gateway.request("run-a", {"method": "resume", "evaluation_id": first["evaluation_id"]})
    assert all(owner == "run-a" for owner, _ in gateway.cancelled)
    gateway.set_run_state("run-a", "enabled")
    assert gateway.request("run-a", {"method": "resume", "evaluation_id": first["evaluation_id"]})["evaluation_id"] == first["evaluation_id"]


def test_key_install_revoke_and_other_keys_preserved(gateway):
    path = gateway.config.authorized_keys
    original = path.read_text()
    assert original.count("restrict,command=") == 2
    assert "--run-id run-a" in original and "-I" in original
    gateway.set_run_state("run-a", "revoked")
    assert "evaluation-run:run-a" not in path.read_text()
    assert "evaluation-run:run-b" in path.read_text()
    with pytest.raises(RequestError, match="unavailable"):
        submit(gateway)
    gateway.set_run_state("run-a", "enabled")
    assert "evaluation-run:run-a" in path.read_text()
    with pytest.raises(RequestError, match="different key"):
        gateway.provision("run-a", public_key(3))
    with pytest.raises(RequestError, match="another run"):
        gateway.provision("run-c", public_key(1))


def fake_record(result, **metadata):
    return {**result, **metadata}


def test_detached_execution_pending_resume_same_state_and_complete_result(gateway):
    job = submit(gateway)
    states = []
    class FakeEvaluator:
        molecules = ["mol"]
        def __init__(self, **config):
            assert config["redis_host"] == "localhost"
            assert config["molecules_dir"] == gateway.config.molecules_dir
        def evaluate(self, spec, *, state_path, task_metadata, should_continue):
            assert spec["kind"] == "source"
            assert task_metadata == {"run_id": "run-a", "evaluation_id": job["evaluation_id"]}
            assert should_continue()
            states.append(state_path)
            return {"status": "pending" if len(states) == 1 else "complete", "results": [],
                    "num_errors": 0, "errors": [], "infrastructure_retries": 1}
    gateway.execute(job["evaluation_id"], FakeEvaluator, fake_record)
    pending = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert pending["status"] == "pending"
    assert pending["num_errors"] == 0
    gateway.request("run-a", {"method": "resume", "evaluation_id": job["evaluation_id"]})
    gateway.execute(job["evaluation_id"], FakeEvaluator, fake_record)
    complete = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert complete["status"] == "complete"
    assert complete["infrastructure_retries"] == 1
    assert states[0] == states[1]
    assert (states[0].parent / "evaluation.json").is_file()
    # Completed evaluations cannot be executed again by duplicate resumes.
    gateway.request("run-a", {"method": "resume", "evaluation_id": job["evaluation_id"]})
    gateway.execute(job["evaluation_id"], FakeEvaluator, fake_record)
    assert len(states) == 2


def test_active_execution_lock_prevents_duplicate_process(gateway):
    job = submit(gateway)
    with gateway.execution_lock(job["evaluation_id"]) as acquired:
        assert acquired
        gateway.execute(job["evaluation_id"], lambda **_: pytest.fail("duplicate evaluator"))


def test_concurrent_duplicate_submits_create_one_job(gateway):
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: submit(gateway), range(12)))
    assert len({r["evaluation_id"] for r in results}) == 1
    with gateway.db() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_pause_during_evaluation_settles_without_killing_running_work(gateway):
    job = submit(gateway)
    other = submit(gateway, "run-b")
    class PausedEvaluator:
        molecules = ["mol"]
        def __init__(self, **_):
            pass
        def evaluate(self, spec, *, should_continue, **_):
            gateway.set_run_state("run-a", "paused")
            assert not should_continue()
            return {"status": "pending", "results": [{"preserved": True}], "num_errors": 0, "errors": []}
    gateway.execute(job["evaluation_id"], PausedEvaluator, fake_record)
    result = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert result["status"] == "cancelled"
    assert result["results"] == [{"preserved": True}]
    assert all(owner == "run-a" for owner, _ in gateway.cancelled)
    assert gateway.request("run-b", {"method": "status", "evaluation_id": other["evaluation_id"]})["status"] == "queued"


def test_cancel_cannot_be_overwritten_by_stale_coordinator_startup(gateway, monkeypatch):
    job = submit(gateway)
    startup_read, release_startup = threading.Event(), threading.Event()
    cancellation_started, cancellation_done = threading.Event(), threading.Event()
    original_db = gateway.db
    class Connection:
        def __init__(self, connection):
            self.connection = connection
        def execute(self, sql, *args):
            name = threading.current_thread().name
            if name == "cancel-startup" and sql == "BEGIN IMMEDIATE":
                cancellation_started.set()
            result = self.connection.execute(sql, *args)
            if name == "execute-startup" and sql == "SELECT state FROM runs WHERE run_id=?":
                startup_read.set()
                assert release_startup.wait(5)
            return result
    @contextmanager
    def db():
        with original_db() as connection:
            yield Connection(connection)
    monkeypatch.setattr(gateway, "db", db)
    seen = []
    class Evaluator:
        molecules = ["mol"]
        def __init__(self, **_):
            pass
        def evaluate(self, *args, should_continue, **kwargs):
            assert cancellation_done.wait(5)
            seen.append(should_continue())
            return {"status": "pending", "results": [], "num_errors": 0, "errors": []}
    errors = []
    def execute():
        try:
            gateway.execute(job["evaluation_id"], Evaluator, fake_record)
        except BaseException as exc:
            errors.append(exc)
    def cancel():
        try:
            gateway.request("run-a", {"method": "cancel", "evaluation_id": job["evaluation_id"]})
        except BaseException as exc:
            errors.append(exc)
        finally:
            cancellation_done.set()
    executor = threading.Thread(target=execute, name="execute-startup")
    canceller = threading.Thread(target=cancel, name="cancel-startup")
    executor.start()
    assert startup_read.wait(5)
    canceller.start()
    try:
        assert cancellation_started.wait(5)
        assert not cancellation_done.wait(0.05), "Cancel was not serialized with startup permission check"
    finally:
        release_startup.set()
        executor.join(5)
        canceller.join(5)
    assert not errors
    assert not executor.is_alive() and not canceller.is_alive()
    assert seen == [False]
    assert gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})["status"] == "cancelled"


def test_cancel_during_final_log_write_is_not_overwritten(gateway, monkeypatch):
    job = submit(gateway)
    from evaluation_access.client import atomic_json
    def cancel_before_commit(path, value):
        if path.name == "evaluation.json":
            result = gateway.request("run-a", {"method": "cancel", "evaluation_id": job["evaluation_id"]})
            assert result["status"] == "cancelling"
        atomic_json(path, value)
    monkeypatch.setattr("evaluation_access.client.atomic_json", cancel_before_commit)
    class Evaluator:
        molecules = ["mol"]
        def __init__(self, **_):
            pass
        def evaluate(self, *args, **kwargs):
            return {"status": "pending", "results": [], "num_errors": 0, "errors": []}
    gateway.execute(job["evaluation_id"], Evaluator, fake_record)
    result = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert result["status"] == "cancelled"
    assert len(gateway.cancelled) >= 2  # initial sweep and final sweep


def test_resume_does_not_hide_a_running_coordinator_crash(gateway):
    job = submit(gateway)
    with gateway.execution_lock(job["evaluation_id"]) as acquired:
        assert acquired
        with gateway.db() as db:
            db.execute("UPDATE jobs SET status='running' WHERE evaluation_id=?", (job["evaluation_id"],))
        response = gateway.request("run-a", {"method": "resume", "evaluation_id": job["evaluation_id"]})
        assert response["status"] == "running"
    # Lock released without a terminal write models an interrupted coordinator.
    result = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert result["status"] == "pending"


def test_unclaimed_queued_job_is_relaunched_with_same_identity(gateway):
    job = submit(gateway)
    launches = len(gateway.launched)
    result = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert len(gateway.launched) == launches + 1
    assert gateway.launched[-1] == job["evaluation_id"]
    assert result["evaluation_id"] == job["evaluation_id"]


def test_record_builder_incomplete_coverage_cannot_be_marked_complete(gateway):
    job = submit(gateway)
    class Evaluator:
        molecules = ["mol"]
        def __init__(self, **_):
            pass
        def evaluate(self, *args, **kwargs):
            return {"status": "complete", "results": [], "num_errors": 0, "errors": []}
    gateway.execute(job["evaluation_id"], Evaluator)
    result = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert result["status"] == "pending"
    assert result["pending_reason"] == "incomplete_molecule_coverage"
    assert result["mean_rel_steps"] is None
    directory = gateway.job_dir(job["evaluation_id"])
    assert "outcome: pending" in (directory / "run.log").read_text()
    assert json.loads((directory / "validate_debug.log").read_text())["status"] == "pending"


def test_scoped_cancel_reads_only_durable_owned_ids(gateway, monkeypatch):
    job = submit(gateway)
    directory = gateway.job_dir(job["evaluation_id"])
    directory.mkdir()
    state_path = directory / "evaluation-state.json"
    state_path.write_text(json.dumps({"records": [{"attempts": [{"task_id": "task-a"}, {"task_id": "task-b"}]}]}))
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["redis_db"] == gateway.config.redis_db
        def withdraw_queued_task(self, task_id, **owner):
            calls.append((task_id, owner))
    monkeypatch.setattr("distributed_validate.client.RemoteOptimizationClient", Client)
    gateway._cancel_tasks(job, state_path)
    assert calls == [(task, {"run_id": "run-a", "evaluation_id": job["evaluation_id"]})
                     for task in ("task-a", "task-b")]


def test_crashed_coordinator_becomes_pending(gateway):
    job = submit(gateway)
    with gateway.db() as db:
        db.execute("UPDATE jobs SET status='running' WHERE evaluation_id=?", (job["evaluation_id"],))
    recovered = gateway.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert recovered["status"] == "pending"
    assert "retained task IDs" in recovered["infrastructure_error"]


def test_subprocess_is_detached_and_trusted(gateway, monkeypatch):
    job = submit(gateway)
    calls = []
    monkeypatch.setattr("evaluation_access.server.subprocess.Popen", lambda *a, **kw: calls.append((a, kw)))
    gateway._launch(job["evaluation_id"])
    argv = calls[0][0][0]
    assert "-I" in argv and "execute" in argv and "--config" in argv
    assert SOURCE not in argv and "shell" not in calls[0][1]
    assert calls[0][1]["start_new_session"] is True


def test_source_normalization_is_static(tmp_path):
    marker = tmp_path / "evil"
    source = f"open({str(marker)!r}, 'w').write('bad')\n" + SOURCE
    spec = normalize_optimizer_spec({"kind": "source", "source": source})
    assert optimizer_cache_key(spec)[0].startswith("source:")
    assert describe_optimizer_spec(spec).startswith("source:")
    assert not marker.exists()
    with pytest.raises(ValueError, match="hash"):
        normalize_optimizer_spec({**spec, "source_sha256": "wrong"})


def test_client_replays_lost_submit_response_without_new_job(gateway, tmp_path):
    program = tmp_path / "algo.py"
    program.write_text(SOURCE)
    state_path = tmp_path / "client.json"
    class InterruptedClient:
        lost = False
        def request(self, request):
            result = gateway.request("run-a", request)
            if request["method"] == "submit" and not self.lost:
                self.lost = True
                raise TransportError("disconnected after accepted submit")
            return result
    client = InterruptedClient()
    with pytest.raises(TransportError):
        evaluate(client, program=program, split="train", state_file=state_path, wait_seconds=0)
    result = evaluate(client, program=program, split="train", state_file=state_path, wait_seconds=0)
    assert result["status"] == "pending"
    with gateway.db() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_cached_receipt_survives_disconnection(gateway, tmp_path):
    program = tmp_path / "algo.py"
    program.write_text(SOURCE)
    state_path = tmp_path / "client.json"
    client = SimpleNamespace(request=lambda req: gateway.request("run-a", req))
    first = evaluate(client, program=program, split="train", state_file=state_path, wait_seconds=0)
    def fail(_):
        raise TransportError("offline")
    result = evaluate(SimpleNamespace(request=fail), program=program, split="train",
                      state_file=state_path, wait_seconds=0)
    assert result["evaluation_id"] == first["evaluation_id"]
    assert result["dataset_release_id"] == "fixture-release"
    assert result["program_id"] == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert result["status"] == "pending"


def test_ssh_uses_no_remote_command_agent_or_control_socket(tmp_path):
    key = tmp_path / "run-key"
    key.write_text("fake private key used only to construct argv")
    argv = ssh_command("user@192.0.2.1", key, known_hosts=tmp_path / "known_hosts")
    assert argv[-1] == "user@192.0.2.1"
    assert "IdentityAgent=none" in argv and "ClearAllForwardings=yes" in argv
    assert "StrictHostKeyChecking=yes" in argv and "ControlPath=none" in argv
    for host in ("-oProxyCommand=sh", "host;id", "host$(id)"):
        with pytest.raises(ValueError):
            ssh_command(host, key)


def test_config_refuses_relative_authority_paths(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"state_root": "../state", "molecules_dir": "/molecules", "release_id": "fixture"}))
    with pytest.raises(ValueError, match="absolute"):
        Config.load(config_path)


@pytest.mark.parametrize("installed_id", ["fixture-release", "older-release", None])
def test_config_pins_installed_dataset_release(tmp_path, installed_id):
    molecules = tmp_path / "molecules"
    molecules.mkdir()
    (molecules / "release.json").write_text(json.dumps({"dataset_release_id": installed_id}))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"state_root": str(tmp_path / "state"),
        "molecules_dir": str(molecules), "release_id": "fixture-release"}))
    if installed_id == "fixture-release":
        assert Config.load(config_path).release_id == installed_id
    else:
        with pytest.raises(ValueError, match="installed dataset release"):
            Config.load(config_path)


def test_gateway_state_cannot_relabel_or_resume_old_dataset_jobs(gateway):
    job = submit(gateway)
    with pytest.raises(RequestError, match="different dataset release"):
        Gateway(replace(gateway.config, release_id="newer-release"), launch=lambda _: pytest.fail("launched"))
    reopened = Gateway(gateway.config, launch=lambda _: None)
    result = reopened.request("run-a", {"method": "results", "evaluation_id": job["evaluation_id"]})
    assert result["release_id"] == "fixture-release"
    assert result["dataset_release_id"] == "fixture-release"


def test_legacy_jobs_without_release_binding_require_new_state_root(gateway):
    submit(gateway)
    with gateway.db() as db:
        db.execute("DELETE FROM gateway_metadata WHERE key='release_id'")
    with pytest.raises(RequestError, match="Legacy gateway state"):
        Gateway(gateway.config)
    with gateway.db() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM gateway_metadata").fetchone()[0] == 0


def test_gateway_rejects_stale_release_before_submitting_any_work(gateway):
    with pytest.raises(RequestError, match="approved dataset release"):
        submit(gateway, expected_release_id="newer-release")
    assert gateway.launched == []
    with gateway.db() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("method", ["submit", "status", "results", "resume", "cancel"])
@pytest.mark.parametrize("change", [
    {"release_id": "older-release", "dataset_release_id": "older-release"},
    {"dataset_release_id": "older-release"},
    {"release_id": None},
])
def test_client_refuses_stale_or_inconsistent_gateway_receipts(monkeypatch, method, change):
    receipt = {"status": "complete", "release_id": "approved-release",
               "dataset_release_id": "approved-release", **change}
    def old_gateway(command, *, input, **kwargs):
        assert json.loads(input)["expected_release_id"] == "approved-release"
        # Model a stale gateway that ignores the requested release.
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "result": receipt}).encode())
    monkeypatch.setattr("evaluation_access.client.subprocess.run", old_gateway)
    with pytest.raises(ValueError, match="approved dataset release"):
        Client(["offline-fixture"], expected_release_id="approved-release").request({"method": method})


def test_client_rejects_gateway_release_change_during_polling(gateway, tmp_path, monkeypatch):
    program = tmp_path / "algo.py"
    program.write_text(SOURCE)
    responses = []
    def gateway_response(command, *, input, **kwargs):
        request = json.loads(input)
        receipt = gateway.request("run-a", request)
        responses.append(request["method"])
        if len(responses) == 3:
            receipt.update(status="complete", release_id="older-release", dataset_release_id="older-release")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "result": receipt}).encode())
    monkeypatch.setattr("evaluation_access.client.subprocess.run", gateway_response)
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="approved dataset release"):
        evaluate(Client(["offline-fixture"], expected_release_id="fixture-release"), program=program,
                 split="train", state_file=tmp_path / "state.json", wait_seconds=1, poll_seconds=.001,
                 output_json=output)
    assert responses == ["submit", "results", "results"]
    assert not output.exists()


@pytest.mark.parametrize("server_checks_release", [False, True])
def test_client_cli_invalidates_old_completed_output_on_release_mismatch(tmp_path, monkeypatch, server_checks_release):
    from evaluation_access.client import main
    key = tmp_path / "key"
    key.write_text("fixture")
    monkeypatch.setenv("EVALUATION_HOST", "fixture")
    monkeypatch.setenv("EVALUATION_KEY", str(key))
    monkeypatch.setenv("EVALUATION_EXPECTED_RELEASE_ID", "approved-release")
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"status": "complete", "evaluation_id": "stale-id"}))
    response = ({"ok": False, "error": "Gateway does not serve the approved dataset release"}
                if server_checks_release else {"ok": True, "result": {
                    "status": "complete", "release_id": "older-release", "dataset_release_id": "older-release"}})
    monkeypatch.setattr("evaluation_access.client.subprocess.run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(response).encode()))
    assert main(["results", "--evaluation-id", "evaluation", "--output-json", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["status"] == "pending"
    assert "approved dataset release" in result["infrastructure_error"]
    assert "evaluation_id" not in result


@pytest.fixture
def client_cli(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("offline fixture")
    program = tmp_path / "algo.py"
    program.write_text(SOURCE)
    monkeypatch.setenv("EVALUATION_HOST", "fixture")
    monkeypatch.setenv("EVALUATION_KEY", str(key))
    monkeypatch.setenv("EVALUATION_EXPECTED_RELEASE_ID", "fixture-release")
    receipt = {"status": "complete", "evaluation_id": "fixture-evaluation", "run_id": "run-a",
               "candidate_commit": "a" * 40, "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
               "split": "train", "release_id": "fixture-release", "dataset_release_id": "fixture-release",
               "mean_rel_steps": .9, "mean_rel_energy": 1.01, "is_valid": 1, "internal_error_count": 0,
               "num_results": 1, "num_errors": 0, "validity_reason": "valid", "infrastructure_retries": 0,
               "progress": {"total": 1, "completed": 1, "pending": 0},
               "results": [{"mol_name": "fixture-molecule", "final_positions": [[1.0, 2.0, 3.0]] * 4096}],
               "errors": [], "recovery_events": [{"task_id": "fixture-task"}]}
    requests = []
    def request(self, value):
        requests.append(value)
        return json.loads(json.dumps(receipt))
    monkeypatch.setattr("evaluation_access.client.Client.request", request)
    monkeypatch.setattr("evaluation_access.client.subprocess.run",
                        lambda *a, **kw: pytest.fail("Unexpected subprocess"))
    return SimpleNamespace(receipt=receipt, requests=requests, output=tmp_path / "result.json",
                           state=tmp_path / "state.json", program=program)


@pytest.mark.parametrize("method", ["evaluate", "results", "status", "resume", "cancel"])
@pytest.mark.parametrize("write_output", [False, True])
def test_client_cli_summarizes_saved_receipt_and_preserves_full_json(client_cli, capsys, method, write_output):
    from evaluation_access.client import main
    fixture = client_cli
    args = [method, "--program", str(fixture.program), "--state-file", str(fixture.state),
            "--candidate-commit", "a" * 40, "--wait-seconds", "0"]
    if method != "evaluate":
        args += ["--evaluation-id", "fixture-evaluation"]
    if write_output:
        args += ["--output-json", str(fixture.output)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    printed = json.loads(captured.out)
    if write_output:
        saved = json.loads(fixture.output.read_text())
        assert saved == {**fixture.receipt, **({"evaluation_log": str(fixture.output)} if method == "evaluate" else {})}
        assert printed["evaluation_log"] == str(fixture.output)
        assert len(captured.out) < 2000
        assert len(fixture.output.read_text()) > 50_000
        assert not {"results", "errors", "recovery_events"} & printed.keys()
        assert "final_positions" not in captured.out
        for field in ("status", "evaluation_id", "candidate_commit", "mean_rel_steps", "mean_rel_energy",
                      "is_valid", "internal_error_count", "num_results", "num_errors", "progress"):
            assert printed[field] == fixture.receipt[field]
    else:
        assert printed == fixture.receipt
        assert not fixture.output.exists()


def test_client_cli_pending_summary_preserves_status_and_exit_code(client_cli, capsys):
    from evaluation_access.client import main
    fixture = client_cli
    fixture.receipt.update(status="running", mean_rel_steps=None, mean_rel_energy=None,
                           is_valid=None, infrastructure_error="fixture worker disconnected")
    assert main(["evaluate", "--program", str(fixture.program), "--state-file", str(fixture.state),
                 "--candidate-commit", "a" * 40, "--wait-seconds", "0", "--output-json", str(fixture.output)]) == 75
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "pending" and printed["service_status"] == "running"
    assert printed["mean_rel_steps"] is None and printed["is_valid"] is None
    assert printed["infrastructure_error"] == "fixture worker disconnected"
    assert json.loads(fixture.output.read_text())["results"] == fixture.receipt["results"]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("commit", ["a" * 40, "b" * 64])
def test_client_cli_valid_commit_is_forwarded(client_cli, monkeypatch, explicit, commit):
    from evaluation_access.client import main
    fixture = client_cli
    fixture.receipt["candidate_commit"] = commit
    git_calls = []
    def git(command, **kwargs):
        git_calls.append(command)
        return SimpleNamespace(returncode=0, stdout=commit + "\n", stderr="")
    monkeypatch.setattr("evaluation_access.client.subprocess.run", git)
    args = ["evaluate", "--program", str(fixture.program), "--state-file", str(fixture.state),
            "--output-json", str(fixture.output)]
    if explicit:
        args += ["--candidate-commit", commit]
    assert main(args) == 0
    assert fixture.requests[0]["candidate_commit"] == commit
    assert git_calls == ([] if explicit else [["git", "rev-parse", "--verify", "HEAD"]])


@pytest.mark.parametrize("explicit,git_result,diagnostic", [
    ("unknown", None, "full hexadecimal Git commit ID"),
    ("", None, "full hexadecimal Git commit ID"),
    ("abc123", None, "full hexadecimal Git commit ID"),
    ("g" * 40, None, "full hexadecimal Git commit ID"),
    (None, (128, "", "fatal: detected dubious ownership in repository at '/workspace'"), "dubious ownership"),
    (None, (0, "", ""), "full hexadecimal Git commit ID"),
    (None, (0, "unknown\n", ""), "full hexadecimal Git commit ID"),
])
def test_client_cli_commit_failure_does_not_submit_or_leave_stale_output(
        client_cli, monkeypatch, capsys, explicit, git_result, diagnostic):
    from evaluation_access.client import main
    fixture = client_cli
    fixture.output.write_text(json.dumps({"status": "complete", "evaluation_id": "stale-id"}))
    if git_result:
        monkeypatch.setattr("evaluation_access.client.subprocess.run", lambda *a, **kw:
            SimpleNamespace(returncode=git_result[0], stdout=git_result[1], stderr=git_result[2]))
    monkeypatch.setattr("evaluation_access.client.ssh_command",
                        lambda *a, **kw: pytest.fail("Transport configured before commit validation"))
    args = ["evaluate", "--program", str(fixture.program), "--state-file", str(fixture.state),
            "--output-json", str(fixture.output)]
    if explicit is not None:
        args += ["--candidate-commit", explicit]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and diagnostic in captured.err
    assert fixture.requests == [] and not fixture.state.exists()
    failed = json.loads(fixture.output.read_text())
    assert failed["status"] == "pending" and diagnostic in failed["infrastructure_error"]
    assert "evaluation_id" not in failed


def test_cached_receipt_cannot_bypass_release_pin_when_offline(gateway, tmp_path):
    program = tmp_path / "algo.py"
    program.write_text(SOURCE)
    state_path = tmp_path / "client.json"
    evaluate(SimpleNamespace(request=lambda req: gateway.request("run-a", req)), program=program,
             split="train", state_file=state_path, wait_seconds=0)
    def offline(_):
        raise TransportError("offline")
    client = SimpleNamespace(request=offline, expected_release_id="newer-release")
    with pytest.raises(ValueError, match="approved dataset release"):
        evaluate(client, program=program, split="train", state_file=state_path, wait_seconds=0,
                 output_json=tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def progress_state(job):
    return {"version": 1, "task_metadata": {"run_id": job["run_id"], "evaluation_id": job["evaluation_id"]},
            "status": "pending", "records": [
                {"mol_name": "done", "attempts": [{"task_id": "a", "state": "complete"}],
                 "payload": {"result": {"mol_name": "done"}, "error": None}},
                {"mol_name": "failed", "attempts": [{"task_id": "b", "state": "complete"}],
                 "payload": {"result": None, "error": "candidate failed", "error_kind": "optimizer_error"}},
                {"mol_name": "retrying", "attempts": [
                    {"task_id": "c", "state": "infrastructure_failed",
                     "failure": {"result": None, "error": "worker lost", "error_kind": "infrastructure_error"}},
                    {"task_id": "d", "state": "prepared"}], "payload": None}]}


def write_progress(gateway, job, state=None):
    directory = gateway.job_dir(job["evaluation_id"])
    directory.mkdir(exist_ok=True)
    path = directory / "evaluation-state.json"
    path.write_text(json.dumps(progress_state(job) if state is None else state))
    return path


def test_admin_progress_counts_terminal_outcomes_and_retries_without_mutating(gateway):
    job = submit(gateway)
    other = submit(gateway, "run-b")
    path = write_progress(gateway, job)
    write_progress(gateway, other)
    before_bytes = path.read_bytes()
    with gateway.db() as db:
        before_jobs = [tuple(row) for row in db.execute("SELECT * FROM jobs ORDER BY evaluation_id")]
    before_launches, before_cancels = list(gateway.launched), list(gateway.cancelled)
    status = gateway.run_status("run-a")
    assert [item["evaluation_id"] for item in status["evaluations"]] == [job["evaluation_id"]]
    evaluation = status["evaluations"][0]
    progress = evaluation["progress"]
    assert {key: progress[key] for key in ("total", "completed", "pending", "attempts", "infrastructure_retries")} == {
        "total": 3, "completed": 2, "pending": 1, "attempts": 4, "infrastructure_retries": 1}
    assert progress["availability"] == "available" and progress["reason"] is None
    assert progress["stale"] is False
    assert "fitness" not in evaluation and "is_valid" not in evaluation
    assert "mol_name" not in json.dumps(progress)
    assert path.read_bytes() == before_bytes
    with gateway.db() as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM jobs ORDER BY evaluation_id")] == before_jobs
    assert gateway.launched == before_launches and gateway.cancelled == before_cancels


def test_progress_exposes_missing_and_stale_snapshot_without_inventing_counts(gateway, monkeypatch):
    job = submit(gateway)
    missing = gateway.run_status("run-a")["evaluations"][0]["progress"]
    assert missing["availability"] == "missing" and missing["reason"] == "state_missing"
    assert all(missing[key] is None for key in ("total", "completed", "pending", "attempts", "stale", "updated_at"))
    path = write_progress(gateway, job)
    os.utime(path, (1_700_000_000, 1_700_000_000))
    monkeypatch.setattr("evaluation_access.server.time.time", lambda: 1_700_000_301)
    stale = gateway.run_status("run-a")["evaluations"][0]["progress"]
    assert stale["availability"] == "available" and stale["stale"] is True
    assert stale["age_seconds"] == 301 and stale["stale_after_seconds"] == 300
    assert stale["updated_at"] == "2023-11-14T22:13:20Z"
    assert stale["completed"] == 2  # old snapshot is retained, not called a failure


@pytest.mark.parametrize("change,reason", [
    (lambda state: state.update(version=True), "invalid_state_version"),
    (lambda state: state["task_metadata"].update(run_id="run-b"), "ownership_mismatch"),
    (lambda state: state.update(records={}), "invalid_records"),
    (lambda state: state["records"][1].update(mol_name="done"), "invalid_molecule_identity"),
    (lambda state: state["records"][2].update(attempts=[]), "invalid_attempts"),
    (lambda state: state["records"][1]["attempts"][0].update(task_id="a"), "invalid_task_identity"),
    (lambda state: state["records"][1]["payload"].update(error_kind="infrastructure_error"), "invalid_terminal_payload"),
    (lambda state: state["records"][1]["payload"].update(error_kind={}), "invalid_terminal_payload"),
    (lambda state: state["records"][0]["payload"]["result"].update(mol_name="wrong"), "invalid_terminal_payload"),
])
def test_invalid_progress_never_returns_partial_counters(gateway, change, reason):
    job = submit(gateway)
    state = progress_state(job)
    change(state)
    write_progress(gateway, job, state)
    progress = gateway.run_status("run-a")["evaluations"][0]["progress"]
    assert progress["availability"] == "invalid" and progress["reason"] == reason
    assert all(progress[key] is None for key in ("total", "completed", "pending", "attempts", "infrastructure_retries"))


def test_progress_read_is_bounded_and_handles_truncated_json(gateway, monkeypatch):
    job = submit(gateway)
    path = write_progress(gateway, job)
    path.write_text('{"version":')
    invalid = gateway.run_status("run-a")["evaluations"][0]["progress"]
    assert invalid["availability"] == "invalid" and invalid["reason"] == "invalid_json"
    monkeypatch.setattr("evaluation_access.server.MAX_PROGRESS_BYTES", 32)
    path.write_text(" " * 33)
    oversized = gateway.run_status("run-a")["evaluations"][0]["progress"]
    assert oversized["availability"] == "oversized" and oversized["completed"] is None


@pytest.mark.parametrize("unsafe_kind", ["file_symlink", "directory_symlink", "parent_symlink", "fifo"])
def test_progress_refuses_symlink_and_nonregular_state(gateway, tmp_path, unsafe_kind):
    job = submit(gateway)
    path = write_progress(gateway, job)
    if unsafe_kind == "file_symlink":
        real = tmp_path / "other-owner-state.json"
        real.write_text(path.read_text())
        path.unlink()
        path.symlink_to(real)
    elif unsafe_kind in {"directory_symlink", "parent_symlink"}:
        directory = path.parent if unsafe_kind == "directory_symlink" else path.parent.parent
        renamed = tmp_path / "other-owner-directory"
        directory.rename(renamed)
        directory.symlink_to(renamed, target_is_directory=True)
    else:
        path.unlink()
        os.mkfifo(path)
    progress = gateway.run_status("run-a")["evaluations"][0]["progress"]
    assert progress["availability"] == "unsafe" and progress["completed"] is None


def test_progress_read_failure_is_explicit(gateway, monkeypatch):
    import errno
    job = submit(gateway)
    write_progress(gateway, job)
    original_open = os.open
    def denied(path, *args, **kwargs):
        if path == "evaluation-state.json":
            raise PermissionError(errno.EACCES, "denied")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr("evaluation_access.server.os.open", denied)
    progress = gateway.run_status("run-a")["evaluations"][0]["progress"]
    assert progress["availability"] == "unreadable" and progress["reason"] == "state_read_failed"
    assert progress["completed"] is None and progress["stale"] is None


def test_status_and_results_share_progress_projection(gateway):
    job = submit(gateway)
    write_progress(gateway, job)
    for method in ("status", "results"):
        result = gateway.request("run-a", {"method": method, "evaluation_id": job["evaluation_id"]})
        assert result["progress"]["completed"] == 2 and result["progress"]["total"] == 3
        assert "fitness" not in result and "is_valid" not in result
