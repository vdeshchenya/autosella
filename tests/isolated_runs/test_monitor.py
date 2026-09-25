from __future__ import annotations

from contextlib import contextmanager
import http.client
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading

import pytest

from isolated_runs.monitor import (Docker, Files, MAX_TAIL, Monitor, MonitorError, Server,
                                   archive_container_log, bounded_command, redact)
from isolated_runs import monitor as monitor_module


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "runs"
    root.mkdir(mode=0o700)
    path = root / "trial-a"
    path.mkdir(mode=0o700)
    for part in ("workspace", "control", "home", "secrets"):
        (path / part).mkdir(mode=0o700)
    state = {"schema": 1, "name": "trial-a", "run_id": "trial-a-" + "a" * 16,
             "status": "running", "purpose": "research", "initial_commit": "b" * 40,
             "created_at": "2026-09-08T09:00:00+00:00",
             "updated_at": "2026-09-08T10:00:00+00:00", "container": "ar-agent-trial-a-" + "a" * 16}
    (path / "control/state.json").write_text(json.dumps(state))
    (path / "control/runtime.json").write_text(json.dumps({"provider": "codex", "model": "gpt-6-astra",
        "effort": "xhigh", "recover_evaluation_ids": ["c" * 32], "auth": {"secret": "must-never-be-served"}}))
    (path / "control/config.json").write_text('{"raw_secret":"must-never-be-served"}')
    return root, path, state


def docker_for(state, *, running=True, text="Ralph iteration 4\n", labels=None):
    calls = []
    def docker(*args, **kwargs):
        calls.append(args)
        if args[0] == "inspect":
            stdout = json.dumps(labels if labels is not None else {"ar.run": state["run_id"]}) + "\n" + json.dumps({
                "Running": running, "Status": "running" if running else "exited", "ExitCode": 0,
                "Error": "PRIVATE metadata must not be served"})
        else:
            stdout = text
        return subprocess.CompletedProcess(args, 0, stdout, "")
    docker.calls = calls
    return docker


@contextmanager
def monitor_for(root, docker):
    monitor = Monitor(root, docker=docker)
    try:
        yield monitor
    finally:
        monitor.close()


def write(path, relative, value):
    target = path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value) if isinstance(value, dict) else value)


def test_progress_projects_metadata_candidate_anchors_recovery_and_real_ralph(run):
    root, path, state = run
    write(path, "workspace/.git/HEAD", "ref: refs/heads/research\n")
    write(path, "workspace/.git/refs/heads/research", "d" * 40)
    write(path, "workspace/.run-ready.json", {"run_id": state["run_id"], "status": "ready"})
    write(path, "workspace/.ralph/ralph-loop.state.json", {"active": True, "iteration": 4, "prompt": "private prompt"})
    for split, identity in (("train", "1"), ("valid", "2")):
        write(path, f"workspace/anchors/{split}.json", {"run_id": state["run_id"], "status": "complete",
            "split": split, "candidate_commit": state["initial_commit"], "evaluation_id": identity * 32,
            "release_id": "release-1", "is_valid": 1, "internal_error_count": 0})
    write(path, "workspace/.evaluation_state/recovered/" + "c" * 32 + ".json",
          {"run_id": state["run_id"], "status": "complete"})
    with monitor_for(root, docker_for(state)) as monitor:
        detail = monitor.detail("trial-a")
        assert detail["current_candidate"] == "d" * 40
        assert detail["anchors_ready"] is True
        assert detail["ralph_iteration"] == 4
        assert detail["execution"].startswith("Ralph running")
        assert detail["recovery"][0]["status"] == "complete"
        assert "not a live gateway poll" in detail["evaluation_source"]
        assert "private prompt" not in json.dumps(detail)
        assert "must-never-be-served" not in json.dumps(detail)
        assert "PRIVATE" not in json.dumps(detail)


def test_verification_is_never_reported_as_ralph_even_with_fake_state(run):
    root, path, state = run
    state["purpose"] = "verification"
    write(path, "control/state.json", state)
    write(path, "workspace/.ralph/ralph-loop.state.json", {"active": True, "iteration": 99})
    write(path, "workspace/verification/provider.log", "real one-shot output")
    with monitor_for(root, docker_for(state)) as monitor:
        detail = monitor.detail("trial-a")
        assert detail["execution"] == "One-shot provider verification; Ralph is not used"
        assert detail["ralph_iteration"] is None
        assert detail["provider_invocations"] == 0
        assert detail["anchors_state"] == "not_started"
        assert detail["container"]["exit_code"] is None
        assert "ralph-state" not in detail["logs"]
        assert "provider" in detail["logs"]
        with pytest.raises(MonitorError):
            monitor.log("trial-a", "ralph-state")


def test_log_allowlist_redacts_all_known_auth_and_never_serves_paths(run):
    root, path, state = run
    key = "this-is-the-exact-run-provider-key"
    token = "this-is-the-exact-oauth-token"
    write(path, "secrets/provider", key)
    write(path, "home/.codex/auth.json", {"tokens": {"access_token": token}})
    write(path, "workspace/.ralph/logs/ralph-123.log", f"iteration\n{key}\n{token}\nAuthorization: Bearer abcdefghijk\n")
    for filename in ("full_log.md", "backlog.md"):
        write(path, "workspace/" + filename, f"# Notes\n{key}\n{token}\nAuthorization: Bearer abcdefghijk\n")
    with monitor_for(root, docker_for(state, text=key + "\n" + token)) as monitor:
        for which in ("ralph/ralph-123.log", "docker", "full-log", "backlog"):
            result = monitor.log("trial-a", which)
            assert key not in result and token not in result
            assert "[REDACTED]" in result
        for which in ("../secrets/provider", "workspace/../secrets/provider", "control/config.json",
                      "home/.codex/auth.json", "/etc/passwd", "ralph/../../control/config.json"):
            with pytest.raises(MonitorError):
                monitor.log("trial-a", which)


@pytest.mark.parametrize("relative", ["workspace", "workspace/.ralph", "workspace/.ralph/logs"])
def test_log_symlink_ancestors_are_never_followed(run, tmp_path, relative):
    root, path, state = run
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ralph-123.log").write_text("secret")
    link = path / relative
    link.parent.mkdir(exist_ok=True, parents=True)
    if link.exists():
        link.rmdir()
    link.symlink_to(outside, target_is_directory=True)
    with monitor_for(root, docker_for(state)) as monitor:
        assert not any(key.startswith("ralph/") for key in monitor.detail("trial-a")["logs"])
        with pytest.raises(MonitorError):
            monitor.log("trial-a", "ralph/ralph-123.log")


def test_no_arbitrary_runs_root_links_leaf_links_or_fifos(run, tmp_path):
    root, path, state = run
    (root / "alias").symlink_to(path, target_is_directory=True)
    (root / "unrecognized").mkdir(mode=0o700)
    write(path, "workspace/.ralph/logs/ralph-123.log", "okay")
    (path / "workspace/run.log").symlink_to(path / "secrets/provider")
    os.mkfifo(path / "workspace/validate_debug.log")
    with monitor_for(root, docker_for(state)) as monitor:
        assert monitor.names() == ["trial-a"]
        assert "run-log" not in monitor.detail("trial-a")["logs"]
        assert "validation-debug" not in monitor.detail("trial-a")["logs"]
        for name in ("alias", "unrecognized", "../trial-a", "/trial-a", "-x"):
            with pytest.raises(MonitorError):
                monitor.detail(name)
    link = tmp_path / "root-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        Monitor(link)
    with pytest.raises(OSError):
        Monitor(link / "trial-a")
    root.chmod(0o755)
    with pytest.raises(MonitorError):
        Monitor(root)


def test_logs_bounded_container_identity_verified_and_metadata_secrets_redacted(run):
    root, path, state = run
    write(path, "workspace/run.log", "x" * (MAX_TAIL * 2))
    write(path, "secrets/provider", "gpt-6-astra")
    docker = docker_for(state, labels={"ar.run": "other-run"})
    with monitor_for(root, docker) as monitor:
        assert len(monitor.log("trial-a", "run-log").encode()) <= MAX_TAIL + 128
        detail = monitor.detail("trial-a")
        assert detail["model"] == "[REDACTED]"
        assert detail["container"]["status"] == "ownership_mismatch"
        assert "docker" not in detail["logs"]
        with pytest.raises(MonitorError):
            monitor.log("trial-a", "docker")
    assert all(args[0] == "inspect" for args in docker.calls)


def test_archive_survives_container_removal_and_refuses_symlink_destination(run, tmp_path):
    root, path, state = run
    write(path, "secrets/provider", "run-api-credential")
    name = archive_container_log(path, state, docker=docker_for(state, text="old stdout run-api-credential\n"))
    stored = path / "operator-logs" / name
    assert stored.stat().st_mode & 0o077 == 0
    assert "run-api-credential" not in stored.read_text()
    absent = lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "not found")
    with monitor_for(root, absent) as monitor:
        assert "archive/" + name in monitor.detail("trial-a")["logs"]
        assert "old stdout" in monitor.log("trial-a", "archive/" + name)
    stored.unlink()
    stored.parent.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    stored.parent.symlink_to(outside)
    with pytest.raises(OSError):
        archive_container_log(path, state, docker=docker_for(state))
    assert not list(outside.iterdir())


def test_archive_error_retains_failure_not_empty_success(run):
    _, path, state = run
    failing = lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "Docker failed")
    with pytest.raises(MonitorError, match="snapshot failed"):
        archive_container_log(path, state, docker=failing)
    assert not (path / "operator-logs").exists()


def test_subprocess_output_and_time_are_bounded_and_remote_context_is_denied(monkeypatch):
    with pytest.raises(MonitorError, match="exceeded"):
        bounded_command([sys.executable, "-c", "print('x'*20000)"], limit=100)
    with pytest.raises(MonitorError, match="timed out"):
        bounded_command([sys.executable, "-c", "import time;time.sleep(10)"], timeout=.1)
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote:2375")
    with pytest.raises(MonitorError, match="remote"):
        Docker()("inspect", "ignored")


def test_http_rebinding_cross_site_and_path_attacks_denied_and_safe_log_content(run):
    root, path, state = run
    write(path, "workspace/run.log", '<script>fetch("https://attacker.invalid")</script>')
    with monitor_for(root, docker_for(state)) as monitor:
        with Server(monitor, 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            authority = "127.0.0.1:" + str(server.server_port)
            def get(route, headers=None, method="GET"):
                client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                client.request(method, route, headers=headers or {})
                response = client.getresponse()
                content = response.read().decode()
                result = response.status, dict(response.getheaders()), content
                client.close()
                return result
            try:
                code, headers, body = get("/")
                assert code == 200 and "Autoresearch progress" in body
                assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
                assert json.loads(get("/api/runs")[2])["gateway_status_enabled"] is False
                monitor.gateway_status = True
                assert json.loads(get("/api/runs")[2])["gateway_status_enabled"] is True
                for headers in ({"Host": "attacker.invalid:" + str(server.server_port)},
                                {"Origin": "https://attacker.invalid"}, {"Origin": "null"},
                                {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"},
                                {"Referer": "https://attacker.invalid/"}):
                    assert get("/api/runs", headers)[0] == 403
                assert get("/api/runs", {"Origin": "http://" + authority})[0] == 200
                for route in ("/control/config.json", "/api/run?name=../trial-a", "/log?name=trial-a&log=../secrets/provider",
                              "/api/run?name=trial-a&name=trial-a", "/api/runs?extra=1"):
                    assert get(route)[0] == 404
                code, headers, body = get("/log?name=trial-a&log=run-log")
                assert code == 200 and headers["Content-Type"].startswith("text/plain")
                assert headers["X-Content-Type-Options"] == "nosniff"
                assert '<script>' in body  # Text is readable but cannot execute.
                assert get("/", method="POST")[0] == 405
            finally:
                server.shutdown()
                worker.join(timeout=2)


def test_auth_container_has_sign_in_phase_without_serving_argv_or_exit_zero(run):
    root, path, state = run
    state["purpose"] = "verification"
    write(path, "control/state.json", state)
    base = docker_for(state)
    def docker(*args, **kwargs):
        result = base(*args, **kwargs)
        result.stdout += '\n["/usr/local/bin/codex"]\n["login", "--device-auth", "do-not-serve-argv"]'
        return result
    with monitor_for(root, docker) as monitor:
        detail = monitor.detail("trial-a")
        assert detail["execution"].startswith("Sign-in in progress")
        assert detail["provider_invocations"] == 0
        assert detail["container"]["exit_code"] is None
        assert "do-not-serve-argv" not in json.dumps(detail)


def test_osc_hyperlink_payload_is_removed_but_label_survives():
    assert redact("go \x1b]8;;https://example.com\x07Example\x1b]8;;\x07 now") == "go Example now"
    assert redact("\x1b]8;;https://example.com\x1b\\Label\x1b]8;;\x1b\\") == "Label"


def test_gateway_status_is_fixed_read_only_argv_projected_and_cached_outside_requests(run):
    root, path, state = run
    write(path, "control/config.json", {"evaluation": {"admin_command": ["trusted-admin", "--private", "value"]}})
    calls = []
    def admin(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps({"run_id": state["run_id"], "state": "paused",
            "private": "DO-NOT-SERVE", "evaluations": [{"run_id": state["run_id"], "evaluation_id": "f" * 32,
                "status": "running", "split": "train", "infrastructure_error": "PRIVATE exception"}]}), "")
    with monitor_for(root, docker_for(state)) as monitor:
        monitor.gateway_status = True
        monitor.admin_runner = admin
        monitor.poll_gateway("trial-a")
        detail = monitor.detail("trial-a")
        assert calls == [(["trusted-admin", "--private", "value", "status", "--run-id", state["run_id"]],
                          {"limit": 2 * 1024 * 1024, "timeout": 10})]
        assert detail["gateway"]["available"] is True and detail["gateway"]["state"] == "paused"
        assert detail["evaluations"][0]["status"] == "running"
        assert "Gateway ledger snapshot at" in detail["evaluation_source"]
        assert "DO-NOT-SERVE" not in json.dumps(detail) and "PRIVATE" not in json.dumps(detail)
        monitor.detail("trial-a")
        assert len(calls) == 1  # HTTP reads cannot dispatch remote commands.
        monitor.admin_runner = lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "private-error", "")
        monitor.poll_gateway("trial-a")
        stale = monitor.detail("trial-a")
        assert stale["gateway"]["available"] is False
        assert stale["gateway"]["enabled"] is True
        assert stale["evaluations"] == detail["evaluations"]
        assert "last successful snapshot" in stale["evaluation_source"]
        assert "gateway poll pending or unavailable" in stale["evaluation_source"]
        assert "private-error" not in json.dumps(stale)


def test_active_anchor_uses_actual_client_receipt_schema_and_matching_gateway_status(run):
    root, path, state = run
    identity = "e8e389a00c52446f99d52cb7eaf99aae"
    receipt = {"evaluation_id": identity, "run_id": state["run_id"], "status": "queued", "split": "train",
        "candidate_commit": state["initial_commit"], "source_sha256": "a" * 64, "release_id": "release-1",
        "timestamp": "2026-09-08T10:01:02Z"}
    write(path, "workspace/.evaluation_state/anchors.json", {"version": 1, "evaluations": {
        "opaque-client-identity": {"request_id": "anchor-train", "evaluation_id": identity, "receipt": receipt}}})
    with monitor_for(root, docker_for(state)) as monitor:
        local = monitor.detail("trial-a")
        assert local["anchors"]["train"]["evaluation_id"] == identity
        assert local["anchors"]["train"]["status"] == "queued"
        assert local["anchors"]["train"]["status_source"] == "saved submission receipt"
        assert local["anchors"]["valid"]["status"] == "not_started"
        assert local["anchors_state"] == "pending_or_invalid"
        assert local["anchors_ready"] is False
        assert local["evaluations"][0]["status"] == "unknown"
        assert local["evaluations"][0]["submission_status"] == "queued"
        assert local["evaluations"][0]["submitted_at"] == "2026-09-08T10:01:02+00:00"
        monitor.gateway_status = True
        monitor.gateway_cache["trial-a"] = {"available": True, "observed_at": "2026-09-08T11:00:00+00:00",
            "state": "enabled", "evaluations": [monitor.evaluation({**receipt, "status": "running"}, state["run_id"])]}
        running = monitor.detail("trial-a")
        assert running["anchors"]["train"]["status"] == "running"
        assert running["anchors"]["train"]["status_source"] == "gateway ledger snapshot"
        monitor.gateway_cache["trial-a"]["evaluations"][0]["status"] = "complete"
        complete_ledger = monitor.detail("trial-a")
        assert complete_ledger["anchors"]["train"]["status"] == "complete"
        assert complete_ledger["anchors"]["train"]["ready"] is False
        assert complete_ledger["anchors_ready"] is False


@pytest.mark.parametrize("mismatch", ["run_id", "candidate_commit", "evaluation_id", "request_id"])
def test_anchor_receipt_must_identify_owned_starting_candidate_and_anchor_request(run, mismatch):
    root, path, state = run
    receipt = {"evaluation_id": "f" * 32, "run_id": state["run_id"], "status": "queued", "split": "train",
               "candidate_commit": state["initial_commit"]}
    entry = {"request_id": "anchor-train", "evaluation_id": "f" * 32, "receipt": receipt}
    if mismatch == "request_id":
        entry[mismatch] = "different-request"
    else:
        receipt[mismatch] = "d" * 40 if mismatch == "candidate_commit" else "other-run" if mismatch == "run_id" else "a" * 32
    write(path, "workspace/.evaluation_state/anchors.json", {"version": 1, "evaluations": {"identity": entry}})
    with monitor_for(root, docker_for(state)) as monitor:
        anchor = monitor.detail("trial-a")["anchors"]["train"]
        assert anchor["evaluation_id"] is None
        assert anchor["ready"] is False


def test_anchor_request_without_acknowledgement_is_submitting(run):
    root, path, state = run
    write(path, "workspace/.evaluation_state/anchors.json", {"version": 1, "evaluations": {
        "identity": {"request_id": "anchor-train"}}})
    with monitor_for(root, docker_for(state)) as monitor:
        detail = monitor.detail("trial-a")
        assert detail["anchors"]["train"]["status"] == "submitting"
        assert detail["anchors"]["train"]["evaluation_id"] is None
        assert detail["anchors"]["valid"]["status"] == "not_started"


def test_validated_local_anchor_is_not_downgraded_by_older_gateway_snapshot(run):
    root, path, state = run
    anchor = {"evaluation_id": "f" * 32, "run_id": state["run_id"], "status": "complete", "split": "train",
        "candidate_commit": state["initial_commit"], "release_id": "release-1", "is_valid": 1, "internal_error_count": 0}
    write(path, "workspace/anchors/train.json", anchor)
    with monitor_for(root, docker_for(state)) as monitor:
        monitor.gateway_status = True
        monitor.gateway_cache["trial-a"] = {"available": True, "observed_at": "2026-09-08T11:00:00+00:00",
            "state": "enabled", "evaluations": [monitor.evaluation({**anchor, "status": "running"}, state["run_id"])]}
        result = monitor.detail("trial-a")["anchors"]["train"]
        assert result["status"] == "complete" and result["ready"] is True


def test_invalid_completed_anchor_explains_existing_evidence_and_keeps_gate_blocked(run):
    root, path, state = run
    anchor = {"evaluation_id": "f" * 32, "run_id": state["run_id"], "status": "complete", "split": "train",
        "candidate_commit": state["initial_commit"], "release_id": "release-1", "is_valid": 0,
        "invalid_reason": "energy_below_baseline", "internal_error_count": 0, "num_results": 457,
        "molecule_count": 457, "mean_rel_energy": 0.9993730539936062,
        "recovery_state_path": "/private/do-not-serve", "errors": ["private diagnostic"]}
    write(path, "workspace/anchors/train.json", anchor)
    with monitor_for(root, docker_for(state)) as monitor:
        monitor.gateway_status = True
        monitor.gateway_cache["trial-a"] = {"available": True, "observed_at": "2026-09-08T11:00:00+00:00",
            "state": "enabled", "evaluations": [monitor.evaluation({**anchor, "status": "running"}, state["run_id"])]}
        result = monitor.detail("trial-a")
        assert result["anchors_state"] == "invalid"
        assert result["gate"] == "Blocked: train anchor invalid (energy_below_baseline)"
        train = result["anchors"]["train"]
        assert train["status"] == "complete" and train["ready"] is False
        assert train["is_valid"] == 0 and train["invalid_reason"] == "energy_below_baseline"
        assert train["mean_rel_energy"] == 0.9993730539936062 and train["internal_error_count"] == 0
        assert "do-not-serve" not in json.dumps(result) and "private diagnostic" not in json.dumps(result)


def test_progress_projects_durable_counts_freshness_and_no_partial_scores(run):
    root, path, state = run
    progress = {"availability": "available", "total": 457, "completed": 411, "pending": 46,
        "attempts": 460, "infrastructure_retries": 3, "updated_at": "2026-09-08T11:00:00+00:00",
        "age_seconds": 315.3, "stale": True, "stale_after_seconds": 300,
        "mean_rel_energy": .9, "path": "/private/do-not-serve"}
    projected = Monitor.evaluation({"run_id": state["run_id"], "evaluation_id": "f" * 32,
        "split": "train", "status": "running", "progress": progress}, state["run_id"])
    assert projected["progress"]["completed"] == 411
    assert projected["progress"]["pending"] == 46
    assert projected["progress"]["stale"] is True
    assert projected["progress"]["updated_at"] == progress["updated_at"]
    assert "mean_rel_energy" not in json.dumps(projected) and "do-not-serve" not in json.dumps(projected)
    for bad in ({**progress, "completed": 500}, {**progress, "pending": -1},
                {**progress, "completed": "411"}, {"availability": {}},
                {"availability": "<script>"}):
        assert Monitor.progress(bad) == {"availability": "invalid"}
    for availability in ("missing", "invalid", "oversized", "unsafe", "unreadable"):
        missing = Monitor.progress({"availability": availability, "total": 457, "completed": 411})
        assert missing["availability"] == availability and "completed" not in missing


def finished_export(run, members, *, manifest=None):
    _, path, state = run
    directory = path / "exports"
    directory.mkdir(exist_ok=True, mode=0o700)
    archive = directory / (state["run_id"] + "-123456.tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        for name, data in members:
            if isinstance(data, tarfile.TarInfo):
                output.addfile(data)
            else:
                data = data.encode() if isinstance(data, str) else data
                info = tarfile.TarInfo(name)
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
        value = manifest if manifest is not None else {"schema": 1, "run_id": state["run_id"],
            "included_diagnostics": [name for name, _ in members]}
        data = json.dumps(value).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        output.addfile(info, io.BytesIO(data))
    archive.chmod(0o600)
    state.update(status="finished", archive={"archive": str(archive),
                 "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    write(path, "control/state.json", state)
    return archive


@pytest.mark.parametrize("stage", ["finishing", "finished"])
def test_finished_logs_only_use_verified_sanitized_export_without_reading_credentials(run, monkeypatch, stage):
    root, path, state = run
    write(path, "workspace/.ralph/logs/ralph-123.log", "unsanitized-token-and-changed-log")
    write(path, "workspace/.ralph/logs/ralph-999.log", "post-finish unvetted log")
    finished_export(run, [("workspace/.ralph/logs/ralph-123.log", "Ralph original [REDACTED]"),
                          ("operator-logs/docker-1.log", "Docker archived [REDACTED]")])
    state["status"] = stage
    write(path, "control/state.json", state)
    def forbidden(*args, **kwargs):
        raise AssertionError("finished view must not read credentials or Docker")
    monkeypatch.setattr(monitor_module, "secrets_for", forbidden)
    with monitor_for(root, forbidden) as monitor:
        detail = monitor.detail("trial-a")
        assert "Verified sanitized export" in detail["log_access"]
        assert "ralph/ralph-123.log" in detail["logs"]
        assert "archive/docker-1.log" in detail["logs"]
        assert "ralph/ralph-999.log" not in detail["logs"]
        result = monitor.log("trial-a", "ralph/ralph-123.log")
        assert "Ralph original [REDACTED]" in result and "unsanitized" not in result
        for key in ("docker", "ralph/ralph-999.log", "control/state.json", "home/.codex/auth.json"):
            with pytest.raises(MonitorError):
                monitor.log("trial-a", key)
        monitor.admin_runner = forbidden
        monitor.poll_gateway("trial-a")


@pytest.mark.parametrize("problem", ["wrong_hash", "wrong_run", "unlisted", "missing", "outside", "symlink"])
def test_finished_export_failure_never_falls_back_to_raw_workspace(run, tmp_path, problem):
    root, path, state = run
    relative = "workspace/.ralph/logs/ralph-123.log"
    write(path, relative, "unredacted-original-secret")
    manifest = {"schema": 1, "run_id": "other-run" if problem == "wrong_run" else state["run_id"],
                "included_diagnostics": [] if problem == "unlisted" else [relative]}
    archive = finished_export(run, [(relative, "sanitized [REDACTED]")], manifest=manifest)
    if problem == "wrong_hash":
        state["archive"]["sha256"] = "0" * 64
    elif problem == "missing":
        archive.unlink()
    elif problem == "outside":
        state["archive"]["archive"] = str(tmp_path / archive.name)
    elif problem == "symlink":
        target = tmp_path / "outside.tar.gz"
        archive.rename(target)
        archive.symlink_to(target)
    write(path, "control/state.json", state)
    with monitor_for(root, docker_for(state)) as monitor:
        detail = monitor.detail("trial-a")
        assert detail["logs"] == []
        assert "unavailable or failed verification" in detail["log_access"]
        with pytest.raises(MonitorError):
            monitor.log("trial-a", "ralph/ralph-123.log")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "duplicate"])
def test_finished_tar_never_follows_members_or_accepts_duplicate_logs(run, kind):
    root, path, state = run
    relative = "workspace/.ralph/logs/ralph-123.log"
    if kind == "duplicate":
        members = [(relative, "first"), (relative, "unvetted duplicate")]
    else:
        member = tarfile.TarInfo(relative)
        member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        member.linkname = "/private/secret"
        members = [(relative, member)]
    finished_export(run, members)
    with monitor_for(root, docker_for(state)) as monitor:
        assert monitor.detail("trial-a")["logs"] == []
        with pytest.raises(MonitorError):
            monitor.log("trial-a", "ralph/ralph-123.log")


def test_finished_export_tails_bounded_caches_verified_data_and_rechecks_changed_archive(run):
    root, path, state = run
    relative = "workspace/.ralph/logs/ralph-123.log"
    archive = finished_export(run, [(relative, b"old\n" * MAX_TAIL + b"last line\n"),
                                   ("home/.codex/auth.json", "MUST NOT SERVE"),
                                   ("../secrets/provider", "MUST NOT SERVE")])
    with monitor_for(root, docker_for(state)) as monitor:
        result = monitor.log("trial-a", "ralph/ralph-123.log")
        assert len(result) < MAX_TAIL + 256 and "last line" in result
        assert len(monitor.export_cache) == 1
        assert monitor.log("trial-a", "ralph/ralph-123.log") == result
        archive.write_bytes(b"changed unvetted contents")
        with pytest.raises(MonitorError):
            monitor.log("trial-a", "ralph/ralph-123.log")


def test_finished_gzip_expansion_and_extended_headers_are_bounded(run, monkeypatch):
    root, path, state = run
    relative = "workspace/.ralph/logs/ralph-123.log"
    archive = finished_export(run, [(relative, b"x" * 20000)])
    assert archive.stat().st_size < 1024
    monkeypatch.setattr(monitor_module, "MAX_EXPORT_BYTES", 1024)
    with monitor_for(root, docker_for(state)) as monitor:
        with pytest.raises(MonitorError):
            monitor.log("trial-a", "ralph/ralph-123.log")
    monkeypatch.setattr(monitor_module, "MAX_EXPORT_BYTES", 1024 * 1024)
    header = tarfile.TarInfo("extended")
    header.type, header.size = tarfile.XHDTYPE, monitor_module.MAX_JSON + 1
    with pytest.raises(MonitorError, match="extended header"):
        monitor_module.ExportTarInfo.frombuf(header.tobuf(), "utf-8", "surrogateescape")


def test_finished_view_reads_actual_launcher_sanitized_diagnostic_export(run, monkeypatch):
    from isolated_runs.launcher import Launcher
    root, path, state = run
    (path / "exports").mkdir(mode=0o700)
    key = "exact-original-provider-token-12345"
    write(path, "secrets/provider", key)
    write(path, "workspace/.ralph/logs/ralph-123.log", "real Ralph output " + key)
    write(path, "workspace/.ralph/ralph-history.json", {"iterations": []})
    write(path, "control/config.json", {"snapshot": {"files": {"algo.py": "a" * 64}}})
    state["status"] = "stopped"
    write(path, "control/state.json", state)
    launcher = Launcher(root)
    monkeypatch.setattr(launcher, "admin", lambda *args, **kwargs: {"run_id": state["run_id"], "evaluations": []})
    exported = launcher.export("trial-a")
    state.update(status="finished", archive=exported)
    write(path, "control/state.json", state)
    (path / "secrets/provider").unlink()
    with monitor_for(root, docker_for(state)) as monitor:
        text = monitor.log("trial-a", "ralph/ralph-123.log")
        assert "real Ralph output [REDACTED]" in text and key not in text
        assert "ralph-history" in monitor.detail("trial-a")["logs"]


def test_finished_verification_uses_exported_invocation_or_reports_unknown(run):
    root, _, state = run
    state["purpose"] = "verification"
    finished_export(run, [("workspace/verification/invocation.json", json.dumps({"run_id": state["run_id"]})),
                          ("workspace/verification/complete.json", '{"status":"verification_complete"}')])
    with monitor_for(root, docker_for(state)) as monitor:
        detail = monitor.detail("trial-a")
        assert detail["provider_invocations"] == 1 and detail["verification_complete"] is True
        assert detail["anchors_state"] == "unavailable"
    finished_export(run, [])
    with monitor_for(root, docker_for(state)) as monitor:
        assert monitor.detail("trial-a")["provider_invocations"] is None


def test_live_display_name_keeps_run_identity_and_routes(run):
    root, path, state = run
    state["display_name"] = "new-data"
    (path / "control/state.json").write_text(json.dumps(state))
    with monitor_for(root, docker_for(state)) as monitor:
        assert monitor.names() == ["trial-a"]
        assert monitor.display_names() == {"trial-a": "new-data"}
        detail = monitor.detail("trial-a")
        assert detail["display_name"] == "new-data"
        assert detail["name"] == "trial-a"
        assert detail["run_id"] == state["run_id"]
        assert detail["container"]["running"] is True
    state["display_name"] = "../invalid"
    (path / "control/state.json").write_text(json.dumps(state))
    with monitor_for(root, docker_for(state)) as monitor:
        assert monitor.display_names() == {"trial-a": "trial-a"}


def test_finished_export_keeps_full_anchor_metadata_above_log_tail_limit(run):
    root, path, state = run
    state["retired_dataset"] = {"release_id": "withdrawn-release"}
    anchor = {"run_id": state["run_id"], "candidate_commit": state["initial_commit"],
              "status": "complete", "is_valid": 1, "internal_error_count": 0,
              "release_id": "withdrawn-release", "results": ["x" * (MAX_TAIL + 1024)]}
    members = []
    for split, identity in [("train", "c" * 32), ("valid", "d" * 32)]:
        members.append((f"workspace/anchors/{split}.json", json.dumps({**anchor, "split": split, "evaluation_id": identity})))
    finished_export(run, members)
    with monitor_for(root, docker_for(state, running=False)) as monitor:
        detail = monitor.detail("trial-a")
        assert detail["anchors_ready"] is True
        assert detail["anchors"]["train"]["evaluation_id"] == "c" * 32
        assert detail["execution"] == "Retired dataset; showing historical results"
