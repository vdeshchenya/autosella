from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import pytest

from isolated_runs.launcher import LaunchError, Launcher, REQUIRED_FILES, load_config, validate_manifest, run_cmd
from isolated_runs.network import BLOCKED_V4, firewall_rules
from isolated_runs.runner import GateError, check_anchor, ralph_command, recover


@pytest.fixture
def config(tmp_path):
    source, data = tmp_path / "approved", tmp_path / "approved-data"
    source.mkdir()
    data.mkdir()
    for name in REQUIRED_FILES:
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# Approved starting file\n")
    (source / "algo.py").write_text("def optimize(system):\n    return system\n")
    # Unapproved source-local state and history must never be copied.
    (source / ".git").mkdir()
    (source / ".git/config").write_text("host-secret-and-old-origin")
    (source / "results.tsv").write_text("old winner 0.002")
    (data / "train.json").write_text("{}")
    (data / "release.json").write_text(json.dumps({"dataset_release_id": "prepared-v1"}))
    credential = tmp_path / "provider-key"
    credential.write_text("this-runs-explicit-provider-key-only")
    credential.chmod(0o600)
    manifest = lambda root, files: {"source": str(root), "files": {
        str(name): hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
    return {"schema": 1, "name": "trial-a", "provider": "codex", "model": "gpt-6-astra", "effort": "high",
            "image": "sha256:" + "a" * 64, "snapshot": manifest(source, REQUIRED_FILES),
            "dataset": manifest(data, ["train.json", "release.json"]), "auth": {"kind": "api_key", "credential_file": str(credential)},
            "evaluation": {"host": "10.1.2.3", "port": 2222, "user": "eval",
                           "host_key": "ssh-ed25519 AAAA", "admin_command": ["gateway-admin"]},
            "resources": {"uid": os.getuid() or 1000, "gid": os.getgid() or 1000,
                          "cpus": 2, "memory": "2g", "pids": 128}}


class FakeLauncher(Launcher):
    def __init__(self, root):
        super().__init__(root)
        self.events = []
        self.containers = {}
        self.evaluations = {}
        self.gateway_states = {}
        self.fail_network = False
        self.fail_pause = False

    def preflight(self, image):
        self.events.append(("preflight", image))

    def admin(self, name, action, *args):
        self.events.append(("admin", name, action, *args))
        if action == "provision":
            return {"authorized_key_installed": True}
        if action == "set-state" and args[-1] == "paused" and self.fail_pause:
            raise LaunchError("offline")
        if action == "status":
            return {"state": self.gateway_states.get(name, "enabled"), "evaluations": self.evaluations.get(name, [])}
        if action == "set-state":
            self.gateway_states[name] = args[-1]
        return {"state": args[-1] if args else "enabled"}

    def docker(self, *args, check=True):
        self.events.append(("docker", *args))
        returncode, output, stderr = 0, "", ""
        if args[0] == "run":
            name = args[args.index("--name") + 1]
            run_id = args[args.index("--label") + 1].removeprefix("ar.run=")
            self.containers[name] = {"Config": {"Labels": {"ar.run": run_id}}, "State": {"Running": True}}
            output = name
        elif args[0] == "inspect":
            name = args[-1]
            if name not in self.containers:
                returncode = 1
                stderr = f"Error: No such object: {name}"
            elif "--format" not in args:
                output = json.dumps([self.containers[name]])
            elif "{{.State.Running}}" in args:
                output = "false" if self.fail_network else "true"
            else:
                output = json.dumps(self.containers[name]["State"])
        elif args[0] == "rm":
            self.containers.pop(args[-1], None)
        elif args[:2] == ("network", "inspect"):
            returncode = 1
        elif args[0] == "logs":
            output = "ISOLATED_NETWORK_READY\n"
        return subprocess.CompletedProcess(args, returncode, output, stderr)


def test_config_fails_closed_and_never_defaults_model(config, tmp_path):
    file = tmp_path / "config.json"
    for field, value in (("model", ""), ("image", "runtime:latest"), ("provider", "other")):
        cfg = copy.deepcopy(config)
        cfg[field] = value
        file.write_text(json.dumps(cfg))
        with pytest.raises(LaunchError):
            load_config(file)
    file.write_text(json.dumps(config))
    assert load_config(file)["model"] == "gpt-6-astra"


@pytest.mark.parametrize("name", [".git/config", ".codex/config.toml", "results.tsv", "non_generalizable.tsv", "backlog.md", "cycle.json", "ideas/old/algo.py", "../secret", "tools/../key", "molecules/test.json"])
def test_manifest_denies_inherited_state(config, name):
    manifest = copy.deepcopy(config["snapshot"])
    manifest["files"] = {name: "a" * 64}
    with pytest.raises(LaunchError):
        validate_manifest(manifest)


def test_snapshot_checks_approval_bytes_and_symlinks(config, tmp_path):
    source = Path(config["snapshot"]["source"])
    (source / "algo.py").write_text("changed after approval")
    with pytest.raises(LaunchError, match="SHA256"):
        validate_manifest(config["snapshot"])
    (source / "algo.py").unlink()
    (source / "algo.py").symlink_to(tmp_path / "provider-key")
    with pytest.raises(LaunchError, match="symlink"):
        validate_manifest(config["snapshot"])


def test_cpu_affinity_survives_config_loading_and_container_creation(config, tmp_path):
    config["resources"]["cpuset_cpus"] = "0,1"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    config = load_config(path)
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    state = launcher.state(config["name"])
    command = launcher.agent_command(config["name"], state, config)
    assert command[command.index("--cpuset-cpus") + 1] == "0,1"
    for value in ["", "0;whoami", 2, "0,,1"]:
        config["resources"]["cpuset_cpus"] = value
        path.write_text(json.dumps(config))
        with pytest.raises(LaunchError, match="cpuset_cpus"):
            load_config(path)


def test_two_created_runs_have_new_git_and_only_owned_mounts(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    second = copy.deepcopy(config)
    second["name"] = "trial-b"
    second["provider"], second["model"], second["effort"] = "claude", "claude-opus-5", "max"
    launcher.create(second)
    a, b = launcher.path("trial-a"), launcher.path("trial-b")
    assert json.loads((a / "control/runtime.json").read_text())["expected_release_id"] == "prepared-v1"
    assert not (a / "workspace/results.tsv").exists()
    assert "host-secret" not in (a / "workspace/.git/config").read_text()
    assert subprocess.run(["git", "-C", str(a / "workspace"), "rev-list", "--all", "--count"],
                          text=True, capture_output=True, check=True).stdout.strip() == "1"
    assert (a / "secrets/evaluation_key").read_bytes() != (b / "secrets/evaluation_key").read_bytes()
    (a / "memory/private.txt").write_text("only a")
    assert not (b / "memory/private.txt").exists()
    for cfg in (config, second):
        state = launcher.state(cfg["name"])
        args = launcher.agent_command(cfg["name"], state, cfg)
        assert "--read-only" in args and "ALL" in args and "no-new-privileges=true" in args
        assert "--privileged" not in args and "NET_ADMIN" not in args
        assert "--user" in args and ":0" not in args[args.index("--user") + 1]
        assert "docker.sock" not in " ".join(args) and "SSH_AUTH_SOCK" not in " ".join(args)
        mounts = [args[i + 1] for i, x in enumerate(args) if x == "--mount"]
        assert all(f"src={launcher.path(cfg['name'])}/" in m for m in mounts)
        assert any("dst=/dataset,readonly" in m for m in mounts)
        assert any("dst=/etc/passwd,readonly" in m for m in mounts)
        assert f'research:x:{cfg["resources"]["uid"]}:{cfg["resources"]["gid"]}:' in (
            launcher.path(cfg["name"]) / "control/passwd").read_text()
        assert not any("host-secret" in x for x in args)


def test_backlog_starts_empty_stays_uncommitted_and_survives_export(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    workspace = launcher.path("trial-a") / "workspace"
    assert (workspace / "backlog.md").read_text() == "# Research backlog\n\nNo deferred ideas yet.\n"
    candidate = workspace / "ideas/rewrite/candidate/algo.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# Deferred implementation\n")
    (workspace / "backlog.md").write_text("# Research backlog\n\nRevisit rewrite when the new solver is available.\n")
    (workspace / "cycle.json").write_text('{"cycle": 2}\n')
    (workspace / ".evaluation_access").mkdir()
    (workspace / ".evaluation_access/client-state.json").write_text('{}\n')
    subprocess.run(["git", "add", "--all"], cwd=workspace, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=workspace, text=True)
    assert staged == ""
    exported = launcher.export("trial-a")
    with tarfile.open(exported["archive"]) as archive:
        assert archive.extractfile("workspace/backlog.md").read() == (workspace / "backlog.md").read_bytes()
        assert archive.extractfile("workspace/ideas/rewrite/candidate/algo.py").read() == candidate.read_bytes()
        assert archive.extractfile("workspace/cycle.json").read() == (workspace / "cycle.json").read_bytes()


def test_lifecycle_pauses_before_stopping_and_recovers_pending_first(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    launcher.start("trial-a")
    launcher.events.clear()
    launcher.stop("trial-a")
    assert launcher.events[0] == ("admin", "trial-a", "set-state", "--state", "paused")
    assert launcher.state("trial-a")["status"] == "stopped"
    launcher.evaluations["trial-a"] = [{"evaluation_id": "own-pending", "status": "pending"},
                                        {"evaluation_id": "own-done", "status": "complete"}]
    launcher.start("trial-a", resume=True)
    runtime = json.loads((launcher.path("trial-a") / "control/runtime.json").read_text())
    assert runtime["recover_evaluation_ids"] == ["own-pending"]
    launcher.stop("trial-a")
    result = launcher.finish("trial-a")
    assert result["key_revoked"] is True
    assert not list((launcher.path("trial-a") / "secrets").iterdir())
    assert any(e == ("admin", "trial-a", "set-state", "--state", "revoked") for e in launcher.events)


def test_network_failure_never_launches_agent(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    launcher.fail_network = True
    with pytest.raises(LaunchError, match="network policy"):
        launcher.start("trial-a")
    assert not any(e[:2] == ("docker", "run") and "--user" in e for e in launcher.events)
    assert launcher.state("trial-a")["status"] == "failed"
    assert not launcher.containers


def test_pause_transport_failure_still_stops_agent_and_blocks_resume(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    launcher.start("trial-a")
    launcher.fail_pause = True
    with pytest.raises(LaunchError, match="pause was not acknowledged"):
        launcher.stop("trial-a")
    assert not launcher.containers
    with pytest.raises(LaunchError):
        launcher.start("trial-a", resume=True)


def test_export_has_no_credential_copies_or_symlink_targets(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    path = launcher.path("trial-a")
    (path / "workspace/disguised.txt").write_bytes((path / "secrets/provider").read_bytes())
    (path / "workspace/secret-link").symlink_to(path / "secrets/evaluation_key")
    (path / "workspace/full_log.md").write_text("This run's observation.")
    oauth = "run-specific-oauth-token-with-enough-entropy"
    (path / "home/.codex/auth.json").write_text(json.dumps({"tokens": {"access_token": oauth}}))
    (path / "workspace/copied-token.txt").write_text(oauth)
    result = launcher.export("trial-a")
    assert Path(result["archive"]).stat().st_mode & 0o077 == 0
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
        assert "workspace/full_log.md" in names
        assert "workspace/disguised.txt" not in names and "workspace/secret-link" not in names
        assert "workspace/copied-token.txt" not in names
        assert not any(".git" in n or "secrets" in n or "control" in n or "home/" in n for n in names)


def test_firewall_blocks_private_metadata_and_ipv6_and_allows_pinned_ssh():
    rules = firewall_rules("10.1.2.3", 2222)
    allow_eval = ["-A", "OUTPUT", "-d", "10.1.2.3", "-p", "tcp", "--dport", "2222", "-j", "ACCEPT"]
    block_private = ["-A", "OUTPUT", "-d", "10.0.0.0/8", "-j", "REJECT"]
    assert rules.index(allow_eval) < rules.index(block_private)
    public_rules = firewall_rules("1.2.3.4", 2222)
    only_ssh = ["-A", "OUTPUT", "-d", "1.2.3.4", "-j", "REJECT"]
    assert public_rules.index(only_ssh) < public_rules.index(["-A", "OUTPUT", "-j", "ACCEPT"])
    assert {"127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.168.0.0/16"} <= set(BLOCKED_V4)
    assert rules[-1] == ["-A", "OUTPUT", "-j", "ACCEPT"]
    with pytest.raises(ValueError):
        firewall_rules("127.0.0.1", 22)


def test_exact_provider_model_settings_pass_through_without_host_bypass():
    cfg = {"provider": "codex", "model": "gpt-6-astra", "effort": "high", "run_id": "trial-a"}
    cmd = ralph_command(cfg)
    assert cmd[cmd.index("--model") + 1] == "gpt-6-astra"
    assert 'model_reasoning_effort="high"' in cmd
    cfg.update(provider="claude", model="claude-opus-5", effort="max")
    cmd = ralph_command(cfg)
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5"
    assert cmd[cmd.index("--effort") + 1] == "max"
    assert "resume" not in cmd  # Fresh CLI sessions; only this run's files persist.



def test_explicit_claude_resume_preserves_session_model_and_effort():
    session = "1c52a6df-2433-4c92-a1bf-58ba8dbb3811"
    cfg = {"provider": "claude", "model": "claude-opus-5", "effort": "max",
           "run_id": "trial-a", "claude_resume_session": session}
    cmd = ralph_command(cfg)
    assert cmd[cmd.index("--resume") + 1] == session
    assert cmd[cmd.index("--model") + 1] == cfg["model"]
    assert cmd[cmd.index("--effort") + 1] == cfg["effort"]
    for invalid in ["", "--help", "not-a-session", 123]:
        with pytest.raises(GateError, match="canonical UUID"):
            ralph_command({**cfg, "claude_resume_session": invalid})


def test_fresh_anchor_gate_rejects_other_runs_and_infrastructure_failures():
    cfg = {"run_id": "trial-a", "initial_commit": "a" * 40, "initial_source_sha256": "b" * 64,
           "expected_release_id": "release"}
    result = {"run_id": "trial-a", "candidate_commit": "a" * 40, "status": "complete", "split": "train",
              "is_valid": 1, "internal_error_count": 0, "evaluation_id": "own-id", "release_id": "release",
              "source_sha256": "b" * 64, "program_id": "b" * 64, "timestamp": "2026-09-08T00:00:00Z",
              "mean_rel_steps": 1.0, "mean_rel_energy": 1.0, "molecule_count": 1,
              "num_results": 1, "num_errors": 0, "errors": [],
              "results": [{"mol_name": "synthetic", "rel_steps": 1.0, "rel_energy": 1.0}]}
    check_anchor(result, "train", cfg)
    for change in ({"run_id": "trial-b"}, {"status": "pending"}, {"internal_error_count": 1}, {"is_valid": 0},
                   {"source_sha256": "c" * 64}, {"program_id": "c" * 64}, {"timestamp": ""},
                   {"release_id": "old-release"}):
        with pytest.raises(GateError):
            check_anchor({**result, **change}, "train", cfg)


@pytest.mark.parametrize("purpose,train_valid,valid_valid,blocked,records", [
    ("verification", 0, 0, False, 0),
    ("verification", 0, 1, False, 0),
    ("verification", 1, 1, False, 0),
    ("research", 0, 1, True, 0),
    ("research", 1, 0, True, 0),
    ("research", 1, 1, False, 1),
])
def test_both_starting_splits_are_observed_before_scientific_eligibility(
        tmp_path, monkeypatch, purpose, train_valid, valid_valid, blocked, records):
    from isolated_runs import runner
    program = tmp_path / "algo.py"
    program.write_text("# fixed approved starting optimizer\n")
    source_hash = hashlib.sha256(program.read_bytes()).hexdigest()
    config = {"run_id": "fresh-run", "initial_commit": "a" * 40, "purpose": purpose,
              "expected_release_id": "prepared-v1"}
    requests, recorded = [], []
    def fake_evaluate(_client, **kwargs):
        split = kwargs["split"]
        requests.append(kwargs)
        observation = {"status": "complete", "split": split, "run_id": "fresh-run",
            "candidate_commit": "a" * 40, "source_sha256": source_hash, "program_id": source_hash,
            "timestamp": "2026-09-08T00:00:00Z", "evaluation_id": "fresh-" + split,
            "release_id": "prepared-v1", "is_valid": train_valid if split == "train" else valid_valid,
            "internal_error_count": 0, "mean_rel_steps": 1.0, "mean_rel_energy": 1.0,
            "molecule_count": 1, "num_results": 1, "num_errors": 0, "errors": [],
            "results": [{"mol_name": "synthetic", "rel_steps": 1.0, "rel_energy": 1.0}]}
        runner.atomic_json(kwargs["output_json"], observation)
        return observation
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: recorded.append(command))
    if blocked:
        with pytest.raises(GateError, match="scientifically invalid"):
            runner.prepare_starting_anchors(object(), config, {}, program=program, workspace=tmp_path)
    else:
        result = runner.prepare_starting_anchors(object(), config, {}, program=program, workspace=tmp_path)
        assert {split: value["is_valid"] for split, value in result.items()} == {"train": train_valid, "valid": valid_valid}
    assert [request["split"] for request in requests] == ["train", "valid"]
    assert [request["request_id"] for request in requests] == ["anchor-train", "anchor-valid"]
    assert all(request["program"] == program and request["wait_seconds"] == 3600 for request in requests)
    assert json.loads((tmp_path / "anchors/train.json").read_text())["is_valid"] == train_valid
    assert json.loads((tmp_path / "anchors/valid.json").read_text())["is_valid"] == valid_valid
    assert len(recorded) == records
    if recorded:
        assert "--starting-anchor" in recorded[0]
    assert not any((tmp_path / filename).exists() for filename in
                   ("results.tsv", "generalizable.tsv", "non_generalizable.tsv"))


@pytest.mark.parametrize("failure", ["internal_error", "pending", "source", "release"])
def test_verification_still_blocks_incomplete_or_untrusted_starting_observations(tmp_path, monkeypatch, failure):
    from isolated_runs import runner
    program = tmp_path / "algo.py"
    program.write_text("# unchanged\n")
    source_hash = hashlib.sha256(program.read_bytes()).hexdigest()
    config = {"run_id": "fresh-run", "initial_commit": "a" * 40, "purpose": "verification",
              "expected_release_id": "prepared-v1"}
    requests = []
    def fake_evaluate(_client, **kwargs):
        split = kwargs["split"]
        requests.append(split)
        result = {"status": "complete", "split": split, "run_id": "fresh-run", "candidate_commit": "a" * 40,
            "source_sha256": source_hash, "program_id": source_hash, "timestamp": "2026-09-08T00:00:00Z",
            "evaluation_id": "fresh-" + split, "release_id": "prepared-v1", "is_valid": 0, "internal_error_count": 0}
        if split == "valid":
            result.update({"internal_error": {"internal_error_count": 1}, "pending": {"status": "pending"},
                           "source": {"source_sha256": "c" * 64}, "release": {"release_id": "different-release"}}[failure])
        runner.atomic_json(kwargs["output_json"], result)
        return result
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not record a scientific decision"))
    with pytest.raises(GateError):
        runner.prepare_starting_anchors(object(), config, {}, program=program, workspace=tmp_path)
    assert requests == ["train", "valid"]
    assert not (tmp_path / "verification/ready.json").exists()


def test_recovery_fails_closed_before_any_new_submission(tmp_path, monkeypatch):
    class Client:
        def __init__(self):
            self.calls = []
        def request(self, request):
            self.calls.append(request)
            return {"status": "pending"}
    client = Client()
    monkeypatch.setattr("isolated_runs.runner.time.sleep", lambda _: None)
    with pytest.raises(GateError):
        recover(client, ["owned-id"], tmp_path)
    assert [x["method"] for x in client.calls] == ["results", "resume", "results"]
    assert all(x["evaluation_id"] == "owned-id" for x in client.calls)


def test_interactive_auth_creation_never_imports_provider_credentials(config, tmp_path):
    config["auth"] = {"kind": "interactive"}
    file = tmp_path / "interactive.json"
    file.write_text(json.dumps(config))
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(load_config(file))
    path = launcher.path("trial-a")
    assert not (path / "secrets/provider").exists()
    assert {p.name for p in (path / "home").iterdir()} == {".codex", ".claude"}
    assert not list((path / "home/.codex").iterdir())
    assert not list((path / "home/.claude").iterdir())
    assert json.loads((path / "control/runtime.json").read_text())["auth_kind"] == "interactive"


def test_hung_admin_is_bounded_and_stop_still_removes_agent(config, tmp_path):
    class HungAdmin(FakeLauncher):
        def admin(self, name, action, *args):
            if action == "set-state" and args[-1] == "paused":
                return run_cmd([sys.executable, "-c", "import time; time.sleep(60)"], timeout=.1)
            return super().admin(name, action, *args)
    launcher = HungAdmin(tmp_path / "runs")
    launcher.create(config)
    launcher.start("trial-a")
    started = time.monotonic()
    with pytest.raises(LaunchError, match="pause was not acknowledged"):
        launcher.stop("trial-a")
    assert time.monotonic() - started < 3
    assert not launcher.containers
    assert launcher.state("trial-a")["status"] == "stopped_unconfirmed"


def test_run_cmd_timeout_kills_descendant_process_group(tmp_path):
    sentinel = tmp_path / "child-survived"
    child = "import time,pathlib,sys;time.sleep(.5);pathlib.Path(sys.argv[1]).write_text('escaped')"
    parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(60)"
    with pytest.raises(LaunchError, match="timed out"):
        run_cmd([sys.executable, "-c", parent, child, str(sentinel)], timeout=.15)
    time.sleep(.6)
    assert not sentinel.exists()


def test_unavailable_docker_never_means_stopped_or_safe_to_export(config, tmp_path):
    class FailedDocker(FakeLauncher):
        offline = False
        def docker(self, *args, check=True):
            if self.offline and args[0] == "inspect":
                return subprocess.CompletedProcess(args, 1, "", "Cannot connect to the Docker daemon")
            return super().docker(*args, check=check)
    launcher = FailedDocker(tmp_path / "runs")
    launcher.create(config)
    launcher.start("trial-a")
    launcher.offline = True
    with pytest.raises(LaunchError, match="Docker container removal"):
        launcher.stop("trial-a")
    assert launcher.containers
    assert launcher.state("trial-a")["status"] == "stopped_unconfirmed"
    assert launcher.public_status("trial-a")["container_state"] == "unknown: Docker unavailable"
    with pytest.raises(LaunchError):
        launcher.export("trial-a")


@pytest.mark.parametrize("agent_directory,filename", [(".codex", "auth.json"), (".claude", ".credentials.json")])
def test_auth_parent_symlink_never_reads_or_deletes_external_files(config, tmp_path, agent_directory, filename):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    outside = tmp_path / "host-auth"
    outside.mkdir()
    canary = outside / filename
    token = "outside-account-token-must-never-be-read"
    canary.write_text(json.dumps({"access_token": token}))
    run = launcher.path("trial-a")
    (run / "home" / agent_directory).rmdir()
    (run / "home" / agent_directory).symlink_to(outside)
    # If the exporter followed the symlink it would incorrectly treat the
    # unrelated outside token as this run's credential and omit this file.
    (run / "workspace/test-observation.txt").write_text(token)
    exported = launcher.export("trial-a")
    assert "workspace/test-observation.txt" not in exported["omitted_credentials"]
    launcher.finish("trial-a")
    assert canary.read_text() == json.dumps({"access_token": token})


def test_finish_retries_cleanup_without_reenabling_revoked_key(config, tmp_path, monkeypatch):
    import isolated_runs.launcher as module
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    original = module.private_unlink
    def fail_once(base, relative):
        raise PermissionError("injected cleanup failure")
    monkeypatch.setattr(module, "private_unlink", fail_once)
    with pytest.raises(PermissionError):
        launcher.finish("trial-a")
    assert launcher.state("trial-a")["status"] == "finishing"
    assert launcher.state("trial-a")["key_revoked"] is True
    launcher.events.clear()
    monkeypatch.setattr(module, "private_unlink", original)
    result = launcher.finish("trial-a")
    assert result["status"] == "finished"
    assert not list((launcher.path("trial-a") / "secrets").iterdir())
    assert not any(event[-1] in {"paused", "enabled"} for event in launcher.events)


def test_network_probe_requires_confirmed_listener():
    from isolated_runs.verify_isolation import confirm_listener
    class NoListener:
        def docker(self, *args):
            return subprocess.CompletedProcess(args, 0, "", "")
    with pytest.raises(LaunchError, match="confirmed listening"):
        confirm_listener(NoListener(), "probe-container")


def test_verification_purpose_propagates_and_optimizer_is_readonly(config, tmp_path):
    file = tmp_path / "config.json"
    file.write_text(json.dumps(config))
    assert load_config(file)["purpose"] == "research"
    config["purpose"] = "verification"
    file.write_text(json.dumps(config))
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(load_config(file))
    state = launcher.state("trial-a")
    assert state["purpose"] == "verification"
    runtime = json.loads((launcher.path("trial-a") / "control/runtime.json").read_text())
    assert runtime["purpose"] == "verification"
    command = launcher.agent_command("trial-a", state, config)
    assert any("dst=/workspace/algo.py,readonly" in a for a in command)
    assert any("dst=/run/verification-permit,readonly" in a for a in command)
    config["purpose"] = "unbounded-smoke"
    file.write_text(json.dumps(config))
    with pytest.raises(LaunchError, match="purpose"):
        load_config(file)


def test_verification_permit_requires_only_fresh_anchors_and_pauses_gateway(config, tmp_path):
    config["purpose"] = "verification"
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    launcher.start("trial-a")
    path, state = launcher.path("trial-a"), launcher.state("trial-a")
    expected_hash = hashlib.sha256((path / "workspace/algo.py").read_bytes()).hexdigest()
    ids = {"train": "fresh-train", "valid": "fresh-valid"}
    (path / "workspace/verification").mkdir()
    (path / "workspace/verification/ready.json").write_text(json.dumps({"run_id": state["run_id"],
        "algo_sha256": expected_hash, "anchor_evaluation_ids": ids, "anchor_validity": {"train": 0, "valid": 1}}))
    jobs = [{"run_id": state["run_id"], "split": split, "evaluation_id": ids[split], "status": "complete",
             "candidate_commit": state["initial_commit"], "source_sha256": expected_hash,
             "release_id": "prepared-v1"} for split in ids]
    (path / "workspace/anchors").mkdir()
    for job in jobs:
        observation = {**job, "program_id": expected_hash, "timestamp": "2026-09-08T00:00:00Z",
                       "is_valid": 0 if job["split"] == "train" else 1, "internal_error_count": 0}
        (path / f'workspace/anchors/{job["split"]}.json').write_text(json.dumps(observation))
    launcher.evaluations["trial-a"] = jobs + [{**jobs[0], "evaluation_id": "unexpected-third"}]
    with pytest.raises(LaunchError, match="exactly the two"):
        launcher.verification_permit("trial-a")
    permit = path / "control/verification-permit/permit.json"
    assert not permit.exists()
    launcher.evaluations["trial-a"] = jobs
    launcher.events.clear()
    assert launcher.verification_permit("trial-a")["gateway_state"] == "paused"
    assert launcher.gateway_states["trial-a"] == "paused"
    assert json.loads(permit.read_text())["anchor_evaluation_ids"] == ids
    assert json.loads(permit.read_text())["anchor_validity"] == {"train": 0, "valid": 1}
    actions = [(e[2], e[-1]) for e in launcher.events if e[0] == "admin"]
    assert actions == [("status", "status"), ("set-state", "paused"), ("status", "status")]


@pytest.mark.parametrize("provider,model,effort", [("codex", "gpt-6-astra", "high"), ("claude", "claude-opus-5", "max")])
def test_verification_is_one_exact_provider_command_with_fixed_finite_task(provider, model, effort):
    from isolated_runs.runner import verification_command, verification_prompt
    cfg = {"provider": provider, "model": model, "effort": effort, "run_id": "own-run"}
    anchors = {split: {"evaluation_id": "own-" + split, "is_valid": 0} for split in ("train", "valid")}
    prompt = verification_prompt(cfg, anchors, "a" * 64)
    command = verification_command(cfg, prompt)
    assert command[0] == ("codex" if provider == "codex" else "claude")
    assert command[command.index("--model") + 1] == model
    assert "ralph" not in command and "--resume" not in command
    assert command.count(prompt) == 1
    assert "primary research source" in prompt and "Do not commit" in prompt
    assert "call evaluate/submit/resume/cancel" in prompt and "gateway is paused" in prompt
    assert "own-train" in prompt and "own-valid" in prompt
    assert "train is_valid=0" in prompt and "Preserve these flags exactly" in prompt
    assert "--json" in command if provider == "codex" else "stream-json" in command


def test_verification_provider_timeout_stops_process_group(tmp_path):
    from isolated_runs.runner import bounded_provider
    sentinel = tmp_path / "child-survived"
    child = "import time,pathlib,sys;time.sleep(.5);pathlib.Path(sys.argv[1]).write_text('escaped')"
    parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(60)"
    with pytest.raises(GateError, match="finite time budget"):
        bounded_provider([sys.executable, "-c", parent, child, str(sentinel)], dict(os.environ),
                         tmp_path / "provider.log", timeout=.15)
    time.sleep(.6)
    assert not sentinel.exists()


def test_verification_requires_matching_trusted_permit_before_provider(tmp_path):
    from isolated_runs.runner import wait_for_verification_permit
    cfg = {"run_id": "own-run"}
    anchors = {split: {"evaluation_id": "own-" + split, "is_valid": 0} for split in ("train", "valid")}
    path = tmp_path / "permit.json"
    with pytest.raises(GateError, match="provider was not started"):
        wait_for_verification_permit(cfg, anchors, "a" * 64, path, wait_seconds=.01)
    path.write_text(json.dumps({"run_id": "other-run", "gateway_state": "paused"}))
    with pytest.raises(GateError, match="does not match"):
        wait_for_verification_permit(cfg, anchors, "a" * 64, path)


def test_verification_records_one_attempt_and_never_invokes_ralph_or_submits(tmp_path, monkeypatch):
    from isolated_runs import runner
    cfg = {"run_id": "own-run", "initial_commit": "a" * 40, "provider": "codex", "model": "gpt-6-astra", "effort": "high",
           "expected_release_id": "prepared-v1"}
    (tmp_path / "algo.py").write_text("# unchanged\n")
    digest = hashlib.sha256((tmp_path / "algo.py").read_bytes()).hexdigest()
    anchors = {split: {"evaluation_id": "own-" + split, "run_id": "own-run", "candidate_commit": "a" * 40,
                      "status": "complete", "split": split, "is_valid": 0, "internal_error_count": 0,
                      "source_sha256": digest, "program_id": digest, "timestamp": "2026-09-08T00:00:00Z",
                      "release_id": "prepared-v1"} for split in ("train", "valid")}
    permit = tmp_path / "permit.json"
    permit.write_text(json.dumps({"run_id": "own-run", "algo_sha256": digest, "gateway_state": "paused",
        "anchor_evaluation_ids": {split: anchors[split]["evaluation_id"] for split in anchors},
        "anchor_validity": {"train": 0, "valid": 0}}))
    invocations = []
    def fake_provider(command, env, log):
        invocations.append(command)
        output = tmp_path / "verification"
        for split in anchors:
            for method in ("status", "results"):
                (output / f"{split}-{method}.json").write_text(json.dumps(anchors[split]))
        (output / "public-source.txt").write_text("Synthetic network response for unit test only.")
        (output / "evidence.json").write_text(json.dumps({"run_id": "own-run", "algo_sha256": digest,
            "provider": "codex", "model": "gpt-6-astra", "effort": "high",
            "anchor_evaluation_ids": {split: anchors[split]["evaluation_id"] for split in anchors},
            "anchor_validity": {"train": 0, "valid": 0},
            "public_source": {"url": "https://arxiv.org/abs/test-fixture", "title": "Synthetic test fixture",
                              "response_file": "verification/public-source.txt"}}))
    class Client:
        def __init__(self):
            self.calls = []
        def request(self, request):
            self.calls.append(request)
            return anchors[request["evaluation_id"].removeprefix("own-")]
    monkeypatch.setattr(runner, "bounded_provider", fake_provider)
    client = Client()
    runner.verify_once(cfg, anchors, {}, client, workspace=tmp_path, permit_path=permit)
    assert len(invocations) == 1 and invocations[0][0] == "codex"
    assert all(call["method"] == "results" for call in client.calls)
    assert json.loads((tmp_path / "verification/complete.json").read_text())["provider_invocations"] == 1
    assert json.loads((tmp_path / "verification/complete.json").read_text())["anchor_validity"] == {"train": 0, "valid": 0}
    assert not (tmp_path / "results.tsv").exists() and not (tmp_path / "generalizable.tsv").exists()
    with pytest.raises(GateError, match="already attempted"):
        runner.verify_once(cfg, anchors, {}, client, workspace=tmp_path, permit_path=permit)
    assert len(invocations) == 1


def test_boundary_probe_cleanup_attempts_both_runs_after_first_cleanup_failure(config, tmp_path):
    from isolated_runs.verify_isolation import verify
    class CleanupFailure(FakeLauncher):
        cleaning = False
        removed = []
        def start_network(self, state, cfg):
            self.cleaning = True
            raise LaunchError("injected startup failure")
        def remove_containers(self, state):
            self.removed.append(state["name"])
            if self.cleaning and state["name"] == "trial-a":
                raise LaunchError("injected first cleanup timeout")
            return super().remove_containers(state)
    launcher = CleanupFailure(tmp_path / "runs")
    launcher.create(config)
    second = copy.deepcopy(config)
    second["name"] = "trial-b"
    launcher.create(second)
    with pytest.raises(LaunchError, match="trial-a: container removal unconfirmed"):
        verify(launcher, ["trial-a", "trial-b"])
    assert launcher.removed[-2:] == ["trial-a", "trial-b"]
    assert not list(launcher.root.glob(".host-canary-*"))


def test_fresh_auth_directory_setup_rejects_external_symlink(tmp_path):
    from isolated_runs.launcher import private_mkdir
    home = tmp_path / "run-home"
    outside = tmp_path / "host-profile"
    home.mkdir()
    outside.mkdir(mode=0o755)
    (home / ".codex").symlink_to(outside)
    with pytest.raises(OSError):
        private_mkdir(home, ".codex", uid=os.getuid(), gid=os.getgid())
    assert outside.stat().st_mode & 0o777 == 0o755
    private_mkdir(home, ".claude", uid=os.getuid(), gid=os.getgid())
    assert (home / ".claude").is_dir()
    assert (home / ".claude").stat().st_mode & 0o777 == 0o700


class RelayLauncher(FakeLauncher):
    fail_relay_stop = False
    fail_relay_probe = False
    fail_relay_supervisor = False
    relay_running = False

    def ensure_relay(self, name):
        self.events.append(("relay-supervisor", "start", self.state(name)["status"]))
        if self.fail_relay_supervisor:
            raise LaunchError("injected relay recovery startup failure")
        return {"name": name, "relay_recovery": "running"}

    def relay(self, state, cfg=None):
        launcher = self
        class Relay:
            def start(self):
                launcher.events.append(("relay", "start"))
                launcher.relay_running = True
            def stop(self):
                launcher.events.append(("relay", "stop"))
                if launcher.fail_relay_stop:
                    raise LaunchError("injected lost relay acknowledgement")
                launcher.relay_running = False
            def status(self):
                return {"state": "running" if launcher.relay_running else "stopped"}
        return Relay()

    def docker(self, *args, check=True):
        result = super().docker(*args, check=check)
        if args[0] == "run" and "-c" in args:
            self.containers.pop(args[args.index("--name") + 1], None)
            output = ('["192.168.65.254"]' if "host.docker.internal" in args[-1]
                      else "" if self.fail_relay_probe else "ISOLATED_RELAY_READY")
            return subprocess.CompletedProcess(args, 0, output, "")
        return result


def with_relay(config):
    config = copy.deepcopy(config)
    config["evaluation"].update(host="192.168.65.254", port=42391,
        relay={"ssh_target": "cpu-33-via-golf", "target_host": "127.0.0.1", "target_port": 22})
    return config


def test_local_relay_uses_pinned_key_and_unchanged_single_endpoint_policy(config, tmp_path):
    config = with_relay(config)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    launcher = RelayLauncher(tmp_path / "runs")
    launcher.create(load_config(config_path))
    path = launcher.path("trial-a")
    runtime = json.loads((path / "control/runtime.json").read_text())
    assert runtime["evaluation"] == {"host": "192.168.65.254", "port": 42391, "user": "eval"}
    assert (path / "secrets/known_hosts").read_text() == "[192.168.65.254]:42391 ssh-ed25519 AAAA\n"
    launcher.start("trial-a")
    probes = [e for e in launcher.events if e[:2] == ("docker", "run") and "-c" in e]
    assert len(probes) == 2 and "--mount" not in probes[0]
    assert all("provider" not in e for e in probes[1])
    assert "Evaluation not found" in probes[1][-1] and "\"method\":\"status\"" in probes[1][-1]
    assert "\"method\":\"submit\"" not in probes[1][-1]
    network = next(e for e in launcher.events if e[:2] == ("docker", "run") and "NET_ADMIN" in e)
    assert network[-2:] == ("192.168.65.254", "42391")
    agent = next(e for e in launcher.events if e[-1] == "/opt/autoresearch/isolated_runs/runner.py")
    assert launcher.events.index(probes[1]) < launcher.events.index(agent)
    assert not any("cpu-33" in str(e) or "/tmp/ar-relay" in str(e) for e in probes + [agent])
    assert launcher.public_status("trial-a")["relay"]["state"] == "running"
    launcher.stop("trial-a")
    assert launcher.public_status("trial-a")["relay"]["state"] == "stopped"
    launcher.start("trial-a", resume=True)
    assert sum(e == ("relay", "start") for e in launcher.events) == 2


def test_relay_pin_mismatch_never_starts_tunnel_or_agent(config, tmp_path):
    config = with_relay(config)
    config["evaluation"]["host"] = "192.168.65.253"
    launcher = RelayLauncher(tmp_path / "runs")
    launcher.create(config)
    with pytest.raises(LaunchError, match="explicitly pinned"):
        launcher.start("trial-a")
    assert ("relay", "start") not in launcher.events
    assert not any(e[-1] == "/opt/autoresearch/isolated_runs/runner.py" for e in launcher.events)


def test_relay_endpoint_failure_stops_tunnel_and_never_starts_research(config, tmp_path):
    launcher = RelayLauncher(tmp_path / "runs")
    launcher.create(with_relay(config))
    launcher.fail_relay_probe = True
    with pytest.raises(LaunchError, match="relay was not verified"):
        launcher.start("trial-a")
    assert not launcher.relay_running and not launcher.containers
    assert not any(e[-1] == "/opt/autoresearch/isolated_runs/runner.py" for e in launcher.events)


def test_unconfirmed_relay_cleanup_blocks_resume_and_export(config, tmp_path):
    launcher = RelayLauncher(tmp_path / "runs")
    launcher.create(with_relay(config))
    launcher.start("trial-a")
    launcher.fail_relay_stop = True
    with pytest.raises(LaunchError, match="SSH relay removal"):
        launcher.stop("trial-a")
    assert not launcher.containers and launcher.state("trial-a")["status"] == "stopped_unconfirmed"
    with pytest.raises(LaunchError):
        launcher.start("trial-a", resume=True)
    with pytest.raises(LaunchError):
        launcher.export("trial-a")
    launcher.fail_relay_stop = False
    launcher.stop("trial-a")
    assert launcher.state("trial-a")["status"] == "stopped"


def test_relay_supervision_starts_after_research_is_running_and_on_resume(config, tmp_path):
    launcher = RelayLauncher(tmp_path / "runs")
    launcher.create(with_relay(config))
    for resume in (False, True):
        launcher.events.clear()
        launcher.start("trial-a", resume=resume)
        supervisor = ("relay-supervisor", "start", "running")
        agent = next(e for e in launcher.events if e[-1] == "/opt/autoresearch/isolated_runs/runner.py")
        assert launcher.events.count(supervisor) == 1
        assert launcher.events.index(agent) < launcher.events.index(supervisor)
        launcher.stop("trial-a")


def test_relay_supervisor_startup_failure_cannot_leave_research_unprotected(config, tmp_path):
    launcher = RelayLauncher(tmp_path / "runs")
    launcher.create(with_relay(config))
    launcher.fail_relay_supervisor = True
    with pytest.raises(LaunchError, match="recovery startup failure"):
        launcher.start("trial-a")
    assert launcher.state("trial-a")["status"] == "failed"
    assert not launcher.containers and not launcher.relay_running
    assert launcher.gateway_states["trial-a"] == "paused"


def test_stop_persists_intent_before_external_actions_even_if_interrupted(config, tmp_path):
    class InterruptedStop(RelayLauncher):
        interrupt_stop = False

        def admin(self, name, action, *args):
            if self.interrupt_stop and action == "set-state" and args[-1] == "paused":
                assert self.state(name)["status"] == "stopping"
                raise KeyboardInterrupt("simulated launcher termination")
            return super().admin(name, action, *args)

    launcher = InterruptedStop(tmp_path / "runs")
    launcher.create(with_relay(config))
    launcher.start("trial-a")
    launcher.interrupt_stop = True
    launcher.events.clear()
    with launcher.lock("trial-a"), pytest.raises(KeyboardInterrupt):
        launcher.stop("trial-a")
    assert launcher.state("trial-a")["status"] == "stopping"
    assert launcher.events == []
    status = launcher.public_status("trial-a")
    assert status["status"] == "stopping" and status["container_state"]["Running"]
    with pytest.raises(LaunchError, match="use start"):
        launcher.start("trial-a", resume=True)
    launcher.interrupt_stop = False
    launcher.stop("trial-a")
    assert launcher.state("trial-a")["status"] == "stopped"
    assert not launcher.relay_running and not launcher.containers


@pytest.mark.parametrize("message", ["error: no such object: own-container",
                                    "error: no such container: own-container",
                                    "error response from daemon: no such container: own-container"])
def test_docker29_lowercase_absence_requires_the_exact_owned_name(message):
    result = subprocess.CompletedProcess(["docker", "inspect"], 1, "[]\n", message + "\n")
    assert Launcher.confirmed_absent(result, "own-container")
    assert not Launcher.confirmed_absent(result, "different-container")
    assert not Launcher.confirmed_absent(subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon"),
                                        "own-container")


def test_auth_relay_cleanup_failure_cannot_leave_a_startable_created_run(config, tmp_path, monkeypatch):
    launcher = RelayLauncher(tmp_path / "runs")
    config = with_relay(config)
    config["auth"] = {"kind": "interactive"}
    launcher.create(config)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    launcher.fail_relay_stop = True
    with pytest.raises(LaunchError, match="SSH relay removal"):
        launcher.auth("trial-a")
    assert launcher.state("trial-a")["status"] == "stopped_unconfirmed"


def test_stop_archives_agent_output_after_stopping_and_before_removal(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    launcher.start("trial-a")
    state = launcher.state("trial-a")
    launcher.events.clear()
    launcher.stop("trial-a")
    stopped = ("docker", "stop", "--time", "5", state["container"])
    archived = ("docker", "logs", "--timestamps", "--tail", "20000", state["container"])
    removed = ("docker", "rm", "--force", state["container"])
    assert launcher.events.index(stopped) < launcher.events.index(archived) < launcher.events.index(removed)
    files = list((launcher.path("trial-a") / "operator-logs").glob("*.log"))
    assert len(files) == 1 and files[0].stat().st_mode & 0o077 == 0


def test_log_archive_failure_retains_container_and_blocks_resume(config, tmp_path, monkeypatch):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    launcher.start("trial-a")
    def unavailable(*args, **kwargs):
        raise OSError("injected archival failure")
    monkeypatch.setattr("isolated_runs.monitor.archive_container_log", unavailable)
    launcher.events.clear()
    with pytest.raises(LaunchError, match="container removal"):
        launcher.stop("trial-a")
    assert launcher.state("trial-a")["status"] == "stopped_unconfirmed"
    assert any(e[:2] == ("docker", "stop") for e in launcher.events)
    assert not any(e[:2] == ("docker", "rm") for e in launcher.events)


def test_export_keeps_vetted_ralph_and_debug_history_with_credentials_redacted(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    run = launcher.path("trial-a")
    provider_key = (run / "secrets/provider").read_text().strip()
    private_key = (run / "secrets/evaluation_key").read_text()
    token = "short-oauth-token"
    (run / "home/.codex/auth.json").write_text(json.dumps({"access_token": token}))
    known = {
        "workspace/.ralph/ralph-loop.state.json": json.dumps({"iteration": 3, "prompt": token}),
        "workspace/.ralph/ralph-history.json": json.dumps({"iterations": [1, 2, 3]}),
        "workspace/.ralph/logs/ralph-123.log": f"iteration 3\n{provider_key}\n{private_key}\n{token}\n",
        "workspace/.ralph/logs/ralph-122.log.1": "earlier iteration\n",
        "workspace/logs/validation.log": "debug result: finished\nAuthorization: Bearer unlisted-token\n",
        "operator-logs/docker-12345.log": f"Docker final output\n{provider_key}\n",
    }
    for relative, text in known.items():
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for relative in ("workspace/.ralph/config.json", "workspace/.ralph/credentials.json",
                     "workspace/.ralph/logs/auth.json", "operator-logs/config.json"):
        (run / relative).write_text("do not export configuration")
    exported = launcher.export("trial-a")
    with tarfile.open(exported["archive"]) as archive:
        names = set(archive.getnames())
        assert known.keys() <= names
        assert not any(name.endswith("config.json") or name.endswith("credentials.json") or name.endswith("auth.json")
                       for name in names)
        manifest = json.load(archive.extractfile("manifest.json"))
        assert set(manifest["included_diagnostics"]) == known.keys()
        assert "workspace/.ralph/ralph-history.json" not in manifest["redacted_diagnostics"]
        output = archive.extractfile("workspace/.ralph/logs/ralph-123.log").read().decode()
        assert "iteration 3" in output and "[REDACTED" in output
        all_bytes = b"\n".join(archive.extractfile(name).read() for name in names)
        assert all(value.encode() not in all_bytes for value in (provider_key, token, private_key, "unlisted-token"))
        assert "Docker final output" in archive.extractfile("operator-logs/docker-12345.log").read().decode()


@pytest.mark.parametrize("relative", ["workspace/.ralph", "workspace/.ralph/logs", "workspace/logs", "operator-logs"])
def test_export_diagnostic_directories_never_follow_external_symlinks(config, tmp_path, relative):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    run = launcher.path("trial-a")
    outside = tmp_path / "outside-logs"
    outside.mkdir()
    for name in ("ralph-123.log", "docker-123.log", "ralph-loop.state.json", "ralph-history.json"):
        (outside / name).write_text("unrelated private data")
    target = run / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    exported = launcher.export("trial-a")
    with tarfile.open(exported["archive"]) as archive:
        assert not any(name.startswith(relative + "/") for name in archive.getnames())
        assert b"unrelated private data" not in b"\n".join(archive.extractfile(name).read() for name in archive.getnames())


def test_finished_export_keeps_saved_redacted_ralph_logs_after_secret_removal(config, tmp_path):
    launcher = FakeLauncher(tmp_path / "runs")
    launcher.create(config)
    run = launcher.path("trial-a")
    key = (run / "secrets/provider").read_text().strip()
    logs = run / "workspace/.ralph/logs"
    logs.mkdir(parents=True)
    (logs / "ralph-123.log").write_text(f"final useful Ralph output\n{key}\n")
    (logs / "ralph-999.log").symlink_to(tmp_path / "provider-key")
    finished = launcher.finish("trial-a")
    assert not list((run / "secrets").iterdir())
    assert launcher.export("trial-a")["archive"] == finished["archive"]
    with tarfile.open(finished["archive"]) as archive:
        output = archive.extractfile("workspace/.ralph/logs/ralph-123.log").read().decode()
        assert "final useful Ralph output" in output and key not in output
        assert "workspace/.ralph/logs/ralph-999.log" not in archive.getnames()
