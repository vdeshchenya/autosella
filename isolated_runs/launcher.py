"""Trusted-host lifecycle controller. Never run this controller inside the agent.

All configuration is explicit JSON, loaded from the operator's private disk.
Subprocess arguments are arrays; no run-controlled string becomes a host shell.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import errno
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time

from isolated_runs.relay import RelayError, SSHRelay, validate_relay
from evaluation_access.release_policy import check_release_supported

SCHEMA = 1
NAME = re.compile(r"[a-z][a-z0-9-]{0,39}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
IMMUTABLE_IMAGE = re.compile(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}\Z")
DENIED_PARTS = {".git", ".ralph", ".codex", ".claude", ".ssh", ".config", ".aws",
    ".cache", ".venv", "__pycache__", "logs", "logs_autoresearch", "memory", "archives",
    "node_modules", "evaluation_access", "isolated_runs"}
DENIED_FILES = {"results.tsv", "generalizable.tsv", "non_generalizable.tsv", "full_log.md", "backlog.md", "cycle.json", "run.log", "auth.json",
    "credentials.json", ".env", "config.json", "config.toml", "settings.json", "settings.local.json"}
REQUIRED_FILES = {"algo.py", "program.md", "ralph_autoresearch_prompt.md", "ralph_autoresearch_template.md",
    "scripts/run_remote_experiment.sh", "scripts/score_log.py", "scripts/record_decision.py", "scripts/cycle.py"}
ROOT = Path(__file__).resolve().parent.parent


class LaunchError(RelayError):
    pass


class LifecycleBusy(LaunchError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(argv: list[str], *, input_text: str | None = None, check: bool = True,
            env: dict | None = None, cwd: Path | None = None, timeout: float = 60,
            capture: bool = True) -> subprocess.CompletedProcess:
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                               stderr=subprocess.PIPE if capture else subprocess.DEVNULL, text=True, env=env, cwd=cwd,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # A wrapper may spawn ssh/grandchildren. Kill the entire new process
        # group, not just the wrapper holding the lifecycle lock.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise LaunchError(f"{Path(argv[0]).name} timed out after {timeout:g}s; service acknowledgement is uncertain") from exc
    p = subprocess.CompletedProcess(argv, process.returncode, stdout or "", stderr or "")
    if check and p.returncode:
        # Never echo command arguments (operator credentials could be in their
        # admin command) or untrusted process output containing key material.
        raise LaunchError(f"{Path(argv[0]).name} failed (exit {p.returncode}); inspect the service locally")
    return p


@contextmanager
def private_parent(base: Path, relative: str):
    """Resolve every component without following run-controlled symlinks."""
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(p in {"..", "."} for p in parts):
        raise LaunchError("invalid private file path")
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, parts[-1]
    finally:
        os.close(fd)


def private_read(base: Path, relative: str) -> bytes | None:
    try:
        with private_parent(base, relative) as (parent, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                return None
            with os.fdopen(fd, "rb") as handle:
                return handle.read()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None
        raise


def private_unlink(base: Path, relative: str) -> None:
    try:
        with private_parent(base, relative) as (parent, name):
            # unlink itself never follows the leaf symlink; parents are opened
            # with O_NOFOLLOW, so no path can leave this run's private directory.
            os.unlink(name, dir_fd=parent)
    except (FileNotFoundError, NotADirectoryError):
        pass
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise


def private_mkdir(base: Path, relative: str, *, uid: int, gid: int) -> None:
    """Create/validate a private directory without traversing any symlink."""
    with private_parent(base, relative) as (parent, name):
        created = False
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
            created = True
        except FileExistsError:
            pass
        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            if created and os.getuid() == 0:
                os.fchown(child, uid, gid)
            os.fchmod(child, 0o700)
        finally:
            os.close(child)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        os.chmod(tmp, 0o600)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(value: str, *, directory: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise LaunchError("configuration paths must be absolute and not symlinks")
    if directory and not path.is_dir():
        raise LaunchError(f"directory missing: {path}")
    if not directory and not path.is_file():
        raise LaunchError(f"file missing: {path}")
    return path.resolve()


def safe_relative(name: str, *, dataset: bool = False) -> PurePosixPath:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p) != name:
        raise LaunchError(f"unsafe manifest path: {name}")
    if any(part.startswith(".") or part in DENIED_PARTS for part in p.parts):
        raise LaunchError(f"state/configuration is forbidden in approved snapshot: {name}")
    if p.name in DENIED_FILES or p.suffix in {".log", ".sqlite", ".db", ".pem", ".key", ".pyc"}:
        raise LaunchError(f"historical/credential file is forbidden: {name}")
    if p.parts[0] == "ideas":
        raise LaunchError(f"preserved research ideas cannot seed a fresh run: {name}")
    if not dataset and p.parts[0] == "molecules":
        raise LaunchError("datasets must use the separate, read-only dataset manifest")
    return p


def validate_manifest(section: dict, *, dataset: bool = False) -> dict:
    if not isinstance(section, dict) or set(section) != {"source", "files"}:
        raise LaunchError("snapshot/dataset must contain only source and explicit files SHA256 mapping")
    source = checked_path(section["source"], directory=True)
    files = section["files"]
    if not isinstance(files, dict) or not files:
        raise LaunchError("empty allowlist is forbidden")
    for name, expected in files.items():
        rel = safe_relative(name, dataset=dataset)
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise LaunchError(f"a full SHA256 is required for {name}")
        path = source / rel
        # Check every component before resolve; symlinks must not silently
        # import another run, a credential, or a host directory.
        if any((source / PurePosixPath(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
            raise LaunchError(f"symlink in approved source: {name}")
        if not path.is_file() or digest(path) != expected:
            raise LaunchError(f"approved SHA256 mismatch or missing file: {name}")
    return {"source": str(source), "files": files}


def approved_release_id(dataset: dict) -> str:
    expected = dataset["files"].get("release.json")
    if expected is None:
        raise LaunchError("approved dataset must include release.json")
    manifest = Path(dataset["source"]) / "release.json"
    if manifest.is_symlink() or not manifest.is_file() or digest(manifest) != expected:
        raise LaunchError("approved release.json hash mismatch or missing file")
    try:
        release_id = json.loads(manifest.read_text())["dataset_release_id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LaunchError("invalid approved release.json") from exc
    if not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", release_id):
        raise LaunchError("invalid approved dataset release identity")
    try:
        check_release_supported(release_id)
    except ValueError as exc:
        raise LaunchError(str(exc)) from exc
    return release_id


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    required = {"schema", "name", "provider", "model", "effort", "image", "snapshot", "dataset",
                "auth", "evaluation", "resources"}
    if set(cfg) - {"purpose"} != required or cfg.get("schema") != SCHEMA:
        raise LaunchError(f"config requires: {', '.join(sorted(required))}; optional: purpose")
    cfg.setdefault("purpose", "research")
    if cfg["purpose"] not in {"research", "verification"}:
        raise LaunchError("purpose must be research or verification")
    if not NAME.fullmatch(cfg["name"]):
        raise LaunchError("name must be 1–40 lowercase letters/digits/hyphens, starting with a letter")
    if cfg["provider"] not in {"codex", "claude"}:
        raise LaunchError("provider must be codex or claude")
    for field in ("model", "effort"):
        if not isinstance(cfg[field], str) or not re.fullmatch(r"[A-Za-z0-9._:/-]+", cfg[field]):
            raise LaunchError(f"explicit {field} is required; no inferred defaults")
    if not IMMUTABLE_IMAGE.fullmatch(cfg["image"]):
        raise LaunchError("image must be an immutable sha256 ID or repository digest")
    cfg["snapshot"] = validate_manifest(cfg["snapshot"])
    cfg["dataset"] = validate_manifest(cfg["dataset"], dataset=True)
    approved_release_id(cfg["dataset"])
    missing = REQUIRED_FILES - cfg["snapshot"]["files"].keys()
    if missing:
        raise LaunchError(f"approved snapshot is missing required files: {sorted(missing)}")
    auth = cfg["auth"]
    if auth == {"kind": "interactive"}:
        pass
    elif isinstance(auth, dict) and set(auth) == {"kind", "credential_file"} and auth["kind"] == "api_key":
        key = checked_path(auth["credential_file"])
        if key.stat().st_mode & 0o077 or not key.read_text().strip():
            raise LaunchError("provider credential file must be nonempty and private (chmod 600)")
        auth["credential_file"] = str(key)
    else:
        raise LaunchError("auth must be {kind: interactive} or {kind: api_key, credential_file: /private/key}")
    ev = cfg["evaluation"]
    if not isinstance(ev, dict) or set(ev) - {"relay"} != {"host", "port", "user", "host_key", "admin_command"}:
        raise LaunchError("evaluation requires host, port, user, host_key, admin_command; optional: relay")
    if "relay" in ev:
        try:
            validate_relay(ev["relay"])
        except RelayError as exc:
            raise LaunchError(str(exc)) from exc
    try:
        ip = ipaddress.IPv4Address(ev["host"])
    except ValueError as exc:
        raise LaunchError("evaluation.host must be a pinned IPv4 address") from exc
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        raise LaunchError("evaluation.host must be reachable outside the agent namespace")
    if type(ev["port"]) is not int or not 1 <= ev["port"] <= 65535:
        raise LaunchError("invalid evaluation port")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", ev["user"]):
        raise LaunchError("invalid evaluation SSH user")
    if not re.fullmatch(r"(?:ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa) [A-Za-z0-9+/=]+", ev["host_key"]):
        raise LaunchError("evaluation.host_key must contain the pinned SSH key type and base64")
    if not isinstance(ev["admin_command"], list) or not ev["admin_command"] or not all(
            isinstance(a, str) and a for a in ev["admin_command"]):
        raise LaunchError("admin_command must be an explicit argv array, never a shell string")
    res = cfg["resources"]
    required_resources = {"cpus", "memory", "pids", "uid", "gid"}
    if not required_resources <= set(res) or set(res) - required_resources - {"cpuset_cpus"}:
        raise LaunchError("resources requires cpus, memory, pids, uid, gid; cpuset_cpus is optional")
    if "cpuset_cpus" in res and (not isinstance(res["cpuset_cpus"], str)
            or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", res["cpuset_cpus"])):
        raise LaunchError("cpuset_cpus must be an explicit Docker CPU list such as 0,1")
    if not isinstance(res["cpus"], (int, float)) or not 0 < res["cpus"] <= 64:
        raise LaunchError("cpus must be in (0, 64]")
    if not re.fullmatch(r"[1-9][0-9]*[mg]", str(res["memory"])):
        raise LaunchError("memory must be an explicit positive limit such as 8g")
    if not all(isinstance(res[f], int) and res[f] > 0 for f in ("uid", "gid", "pids")):
        raise LaunchError("uid/gid must be non-root; pids must be a positive limit")
    if os.getuid() != 0 and (res["uid"] != os.getuid() or res["gid"] != os.getgid()):
        raise LaunchError("use the local operator uid/gid, or run trusted provisioning as root")
    return cfg


def copy_manifest(section: dict, destination: Path) -> None:
    # Validate a second time at use, and hash bytes actually written to close
    # stale approval races. A changing source fails without launching anything.
    for name, expected in section["files"].items():
        src = Path(section["source"]) / name
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise LaunchError(f"snapshot changed during creation: {name}")
        dst = destination / name
        dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        dst.write_bytes(data)
        dst.chmod(0o700 if src.stat().st_mode & stat.S_IXUSR else 0o600)


def git_env(home: Path) -> dict:
    # Do not inherit Git alternate-object directories, templates, global
    # includes, identity, hooks, SSH commands, or host configuration.
    return {"PATH": os.defpath + ":/usr/local/bin:/opt/homebrew/bin", "HOME": str(home),
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}


class Launcher:
    def __init__(self, root: Path):
        self.root = root.expanduser().absolute()
        if self.root.is_symlink():
            raise LaunchError("run root must not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_mode & 0o077:
            raise LaunchError("run root must be private (chmod 700)")

    def path(self, name: str) -> Path:
        if not NAME.fullmatch(name):
            raise LaunchError("invalid run name")
        result = self.root / name
        if result.is_symlink():
            raise LaunchError("run directory must not be a symlink")
        return result

    @contextmanager
    def lock(self, name: str):
        lock = self.root / ("." + name + ".lock")
        with lock.open("a") as handle:
            os.chmod(lock, 0o600)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LifecycleBusy("another lifecycle action owns this run") from exc
            yield

    def state(self, name: str) -> dict:
        return json.loads((self.path(name) / "control/state.json").read_text())

    def save(self, name: str, state: dict) -> None:
        state["updated_at"] = now()
        write_json(self.path(name) / "control/state.json", state)

    def docker(self, *args: str, check: bool = True):
        return run_cmd(["docker", *args], check=check, timeout=30)

    def preflight(self, image: str) -> None:
        if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
            raise LaunchError("run the controller on the Docker host; remote Docker environment overrides are forbidden")
        context = json.loads(self.docker("context", "inspect").stdout)
        endpoint = context[0]["Endpoints"]["docker"]["Host"]
        if not endpoint.startswith("unix://"):
            raise LaunchError("remote Docker contexts are forbidden; run the controller on that host over SSH")
        info = json.loads(self.docker("info", "--format", "{{json .}}").stdout)
        if info.get("OSType") != "linux":
            raise LaunchError("a Linux Docker engine is required")
        spec = json.loads(self.docker("image", "inspect", image).stdout)[0]
        if spec.get("Config", {}).get("Labels", {}).get("org.gigaopt.isolated-runs") != str(SCHEMA):
            raise LaunchError("image lacks the isolated-runs runtime label; build with the provided installer")

    def admin(self, name: str, action: str, *args: str) -> dict:
        state = self.state(name)
        cfg = json.loads((self.path(name) / "control/config.json").read_text())
        p = run_cmd([*cfg["evaluation"]["admin_command"], action, "--run-id", state["run_id"], *args], timeout=30)
        try:
            result = json.loads(p.stdout)
        except ValueError as exc:
            raise LaunchError("evaluation admin did not return a JSON acknowledgement") from exc
        if not isinstance(result, dict) or result.get("ok") is False:
            raise LaunchError("evaluation admin rejected the lifecycle action")
        if result.get("run_id") != state["run_id"]:
            raise LaunchError("evaluation admin acknowledgement belongs to a different run")
        if action == "status" and not isinstance(result.get("evaluations"), list):
            raise LaunchError("evaluation admin status lacks the owned evaluation list")
        if action == "set-state" and result.get("state") != args[-1]:
            raise LaunchError("evaluation admin did not acknowledge the requested state")
        return result

    def create(self, cfg: dict) -> dict:
        name = cfg["name"]
        path = self.path(name)
        if path.exists():
            raise LaunchError("run already exists; use status/resume or choose a new name")
        self.preflight(cfg["image"])
        # A UUID-like suffix is never reused even if two hosts use the same
        # human-readable name. Names alone are not an authorization boundary.
        run_id = name + "-" + os.urandom(8).hex()
        path.mkdir(mode=0o700)
        for part in ("workspace", "home", "scratch", "memory", "tools", "dataset", "secrets", "control", "exports"):
            (path / part).mkdir(mode=0o700)
        for directory in (".codex", ".claude"):
            private_mkdir(path / "home", directory, uid=cfg["resources"]["uid"], gid=cfg["resources"]["gid"])
        state = {"schema": SCHEMA, "name": name, "run_id": run_id, "status": "creating", "created_at": now(),
                 "purpose": cfg.get("purpose", "research"),
                 "container": "ar-agent-" + run_id, "network_container": "ar-net-" + run_id,
                 "network": "ar-" + run_id, "image": cfg["image"], "key_revoked": False}
        self.save(name, state)
        write_json(path / "control/config.json", cfg)
        try:
            copy_manifest(cfg["snapshot"], path / "workspace")
            copy_manifest(cfg["dataset"], path / "dataset")
            expected_release_id = approved_release_id({"source": str(path / "dataset"),
                                                       "files": cfg["dataset"]["files"]})
            clean_env = git_env(path / "home")
            run_cmd(["git", "init", "--template=", "--initial-branch=research", str(path / "workspace")], env=clean_env)
            for key, value in (("user.name", "Autoresearch"), ("user.email", "run@isolated.invalid"),
                               ("core.hooksPath", "/dev/null")):
                run_cmd(["git", "config", "--local", key, value], env=clean_env, cwd=path / "workspace")
            run_cmd(["git", "add", "--all"], env=clean_env, cwd=path / "workspace")
            run_cmd(["git", "-c", "commit.gpgsign=false", "commit", "-m", "Approved starting snapshot"],
                    env=clean_env, cwd=path / "workspace")
            state["initial_commit"] = run_cmd(["git", "rev-parse", "HEAD"], env=clean_env, cwd=path / "workspace").stdout.strip()
            # Research notes and rejected implementations are private artifacts,
            # initialized after the approved root commit rather than inherited.
            (path / "workspace/.git/info").mkdir(exist_ok=True)
            with (path / "workspace/.git/info/exclude").open("a") as handle:
                handle.write("\n/backlog.md\n/ideas/\n/full_log.md\n/results.tsv\n"
                             "/generalizable.tsv\n/non_generalizable.tsv\n"
                             "/evaluation_results/\n/diagnostics/\n/anchors/\n/.evaluation_state/\n/.evaluation_access/\n/cycle.json\n")
            (path / "workspace/backlog.md").write_text("# Research backlog\n\nNo deferred ideas yet.\n")
            (path / "workspace/ideas").mkdir()
            run_cmd(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", run_id,
                     "-f", str(path / "secrets/evaluation_key")])
            if cfg["auth"]["kind"] == "api_key":
                (path / "secrets/provider").write_bytes(Path(cfg["auth"]["credential_file"]).read_bytes())
                (path / "secrets/provider").chmod(0o600)
            ev = cfg["evaluation"]
            host = ev["host"] if ev["port"] == 22 else f'[{ev["host"]}]:{ev["port"]}'
            (path / "secrets/known_hosts").write_text(f'{host} {ev["host_key"]}\n')
            (path / "control/resolv.conf").write_text("nameserver 1.1.1.1\nnameserver 8.8.8.8\n")
            # Numeric host UIDs may not exist in a generic image. Give CLI and
            # package-manager NSS lookups only this run's explicit identity.
            res = cfg["resources"]
            (path / "control/passwd").write_text("root:x:0:0:root:/root:/bin/sh\n"
                f'research:x:{res["uid"]}:{res["gid"]}:Research:/home/research:/bin/bash\n')
            (path / "control/group").write_text(f'root:x:0:\nresearch:x:{res["gid"]}:research\n')
            (path / "control/starting-algo.py").write_bytes((path / "workspace/algo.py").read_bytes())
            runtime = {"schema": SCHEMA, "run_id": run_id, "initial_commit": state["initial_commit"],
                       "expected_release_id": expected_release_id,
                       "purpose": cfg.get("purpose", "research"),
                       "provider": cfg["provider"], "model": cfg["model"], "auth_kind": cfg["auth"]["kind"],
                       "effort": cfg["effort"], "evaluation": {k: ev[k] for k in ("host", "port", "user")}}
            write_json(path / "control/runtime.json", runtime)
            (path / "control/verification-permit").mkdir(mode=0o755)
            for item in ("runtime.json", "starting-algo.py", "resolv.conf", "passwd", "group"):
                (path / "control" / item).chmod(0o644)
            self.save(name, state)
            # The private key never leaves this run directory. The trusted
            # admin command receives only the public half, installs it with a
            # forced command, and scopes its server-side ledger to run_id.
            provision = self.admin(name, "provision", "--public-key-text", (path / "secrets/evaluation_key.pub").read_text().strip())
            if provision.get("authorized_key_installed") is not True:
                raise LaunchError("gateway must install the forced per-run key automatically; configure authorized_keys")
            if os.getuid() == 0:
                for part in ("workspace", "home", "scratch", "memory", "tools", "dataset", "secrets"):
                    for item in [path / part, *(path / part).rglob("*")]:
                        os.chown(item, cfg["resources"]["uid"], cfg["resources"]["gid"])
            state["status"] = "created"
            self.save(name, state)
            return self.public_status(name)
        except Exception:
            state["status"] = "creation_failed"
            self.save(name, state)
            # Provision can succeed while its acknowledgement is lost. Always
            # try revocation; retain key and state if its success is uncertain.
            try:
                self.admin(name, "set-state", "--state", "revoked")
                state["key_revoked"] = True
                self.save(name, state)
            except Exception:
                pass
            raise

    def mount(self, source: Path, target: str, *, readonly: bool = False) -> list[str]:
        if "," in str(source):
            raise LaunchError("Docker bind source paths must not contain commas")
        return ["--mount", f'type=bind,src={source},dst={target}' + (",readonly" if readonly else "")]

    def agent_command(self, name: str, state: dict, cfg: dict) -> list[str]:
        path, res = self.path(name), cfg["resources"]
        args = ["run", "--detach", "--name", state["container"], "--label", f'ar.run={state["run_id"]}',
                "--network", "container:" + state["network_container"], "--user", f'{res["uid"]}:{res["gid"]}',
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
                "--pids-limit", str(res["pids"]), "--cpus", str(res["cpus"]), "--memory", res["memory"],
                "--memory-swap", res["memory"], "--init", "--workdir", "/workspace",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m", "--tmpfs", "/run:rw,nosuid,nodev,size=16m",
                "--log-driver", "local", "--log-opt", "max-size=10m", "--log-opt", "max-file=3"]
        if res.get("cpuset_cpus"):
            args += ["--cpuset-cpus", res["cpuset_cpus"]]
        for part, target in (("workspace", "/workspace"), ("home", "/home/research"),
                             ("scratch", "/scratch"), ("memory", "/memory"), ("tools", "/tools")):
            args += self.mount(path / part, target)
        for part, target in (("dataset", "/dataset"), ("secrets", "/run/secrets"),
                             ("control/runtime.json", "/etc/autoresearch.json"),
                             ("control/starting-algo.py", "/etc/starting-algo.py"),
                             ("control/resolv.conf", "/etc/resolv.conf"),
                             ("control/passwd", "/etc/passwd"), ("control/group", "/etc/group")):
            args += self.mount(path / part, target, readonly=True)
        if cfg.get("purpose", "research") == "verification":
            args += self.mount(path / "control/starting-algo.py", "/workspace/algo.py", readonly=True)
            args += self.mount(path / "control/verification-permit", "/run/verification-permit", readonly=True)
        args += ["--env", "HOME=/home/research", "--env", "CODEX_HOME=/home/research/.codex",
                 "--env", "CLAUDE_CONFIG_DIR=/home/research/.claude", "--env", "XDG_CONFIG_HOME=/home/research/.config",
                 "--env", "XDG_CACHE_HOME=/home/research/.cache", "--env", "TMPDIR=/scratch",
                 "--env", "JAX_ENABLE_X64=true", "--env", "PYTHONPATH=/opt/autoresearch",
                 "--env", "PATH=/tools/bin:/home/research/.local/bin:/usr/local/bin:/usr/bin:/bin",
                 "--entrypoint", "python3", cfg["image"], "/opt/autoresearch/isolated_runs/runner.py"]
        return args

    def start(self, name: str, *, resume: bool = False) -> dict:
        state = self.state(name)
        if state["status"] == "running":
            raise LaunchError("run is already marked running; inspect status before changing it")
        if state["status"] not in ({"stopped", "starting", "failed"} if resume else {"created"}):
            raise LaunchError("use start for a created run, resume for a stopped/interrupted run")
        cfg = json.loads((self.path(name) / "control/config.json").read_text())
        if cfg["auth"]["kind"] == "interactive" and not state.get("authenticated_at"):
            raise LaunchError("complete this run's interactive auth operation before starting research")
        runtime_path = self.path(name) / "control/runtime.json"
        runtime = json.loads(runtime_path.read_text())
        try:
            check_release_supported(runtime.get("expected_release_id"))
        except ValueError as exc:
            raise LaunchError(str(exc)) from exc
        self.preflight(cfg["image"])
        # Reconcile state first; the in-container gate recovers owned pending
        # evaluations and both anchors before allowing any new research.
        self.admin(name, "set-state", "--state", "enabled")
        remote = self.admin(name, "status")
        runtime["recover_evaluation_ids"] = [job["evaluation_id"] for job in remote.get("evaluations", [])
                                              if job["status"] != "complete"]
        write_json(runtime_path, runtime)
        runtime_path.chmod(0o644)
        state["status"] = "starting"
        self.save(name, state)
        (self.path(name) / "workspace/.run-ready.json").unlink(missing_ok=True)
        (self.path(name) / "control/verification-permit/permit.json").unlink(missing_ok=True)
        try:
            self.remove_containers(state)
            self.start_network(state, cfg)
            self.docker(*self.agent_command(name, state, cfg))
            state["status"] = "running"
            self.save(name, state)
            if "relay" in cfg["evaluation"]:
                self.ensure_relay(name)
            return self.public_status(name)
        except Exception:
            state["status"] = "failed"
            self.save(name, state)
            try:
                self.stop(name)
                state["status"] = "failed"
                self.save(name, state)
            except Exception:
                pass  # stop persisted the unconfirmed state for operator retry.
            raise

    def start_network(self, state: dict, cfg: dict) -> None:
        if "relay" in cfg["evaluation"]:
            self.verify_relay_host(state, cfg)
            self.relay(state, cfg).start()
        if self.docker("network", "inspect", state["network"], check=False).returncode:
            self.docker("network", "create", "--driver", "bridge", "--opt",
                        "com.docker.network.bridge.enable_icc=false", "--label", f'ar.run={state["run_id"]}', state["network"])
        self.docker("run", "--detach", "--name", state["network_container"], "--label", f'ar.run={state["run_id"]}',
                "--network", state["network"], "--read-only", "--cap-drop", "ALL", "--cap-add", "NET_ADMIN",
                "--cap-add", "SETUID", "--cap-add", "SETGID", "--cap-add", "SETPCAP",
                "--security-opt", "no-new-privileges=true", "--pids-limit", "16", "--memory", "64m", "--cpus", "0.1",
                "--sysctl", "net.ipv6.conf.all.disable_ipv6=1", "--tmpfs", "/run:rw,nosuid,nodev,size=8m",
                "--entrypoint", "python3", cfg["image"], "/opt/autoresearch/isolated_runs/network.py",
                cfg["evaluation"]["host"], str(cfg["evaluation"]["port"]))
        for _ in range(30):
            logs = self.docker("logs", state["network_container"]).stdout
            alive = self.docker("inspect", "--format", "{{.State.Running}}", state["network_container"]).stdout.strip()
            if alive != "true":
                raise LaunchError("network policy helper failed; agent was not started")
            if "ISOLATED_NETWORK_READY" in logs:
                if "relay" in cfg["evaluation"]:
                    self.verify_relay_endpoint(state, cfg)
                return
            time.sleep(0.2)
        raise LaunchError("network policy was not acknowledged; agent was not started")

    def relay(self, state: dict, cfg: dict | None = None) -> SSHRelay:
        if cfg is None:
            cfg = json.loads((self.path(state["name"]) / "control/config.json").read_text())
        return SSHRelay(self.path(state["name"]) / "control", state["run_id"], cfg["evaluation"], run_cmd)

    def stop_relay(self, state: dict) -> None:
        cfg = json.loads((self.path(state["name"]) / "control/config.json").read_text())
        if "relay" in cfg["evaluation"]:
            self.relay(state, cfg).stop()

    def ensure_relay(self, name: str) -> dict:
        """Enable recovery for a running run, adopting its healthy relay."""
        from isolated_runs.relay_recovery import ensure_supervisor
        ensure_supervisor(self, name)
        return {"name": name, "relay_recovery": "running"}

    def relay_probe_command(self, state: dict, cfg: dict, script: str, *, protected: bool) -> list[str]:
        res = cfg["resources"]
        command = ["run", "--rm", "--name", state["container"], "--label", f'ar.run={state["run_id"]}',
                   "--user", f'{res["uid"]}:{res["gid"]}', "--read-only", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges=true", "--pids-limit", "16", "--memory", "64m",
                   "--cpus", "0.1", "--network", "container:" + state["network_container"] if protected else "bridge"]
        if protected:
            for name in ("evaluation_key", "known_hosts"):
                command += self.mount(self.path(state["name"]) / "secrets" / name, "/run/secrets/" + name,
                                      readonly=True)
            for name in ("passwd", "group"):
                command += self.mount(self.path(state["name"]) / "control" / name, "/etc/" + name, readonly=True)
        return command + ["--entrypoint", "python3", cfg["image"], "-I", "-c", script]

    def verify_relay_host(self, state: dict, cfg: dict) -> None:
        # Docker's host name is resolved by a trusted no-mount probe before
        # installing the firewall. The agent never needs Docker's embedded DNS.
        script = ('import json,socket; print(json.dumps(sorted({x[4][0] for x in '
                  'socket.getaddrinfo("host.docker.internal",None,socket.AF_INET)})))')
        result = self.docker(*self.relay_probe_command(state, cfg, script, protected=False))
        if json.loads(result.stdout) != [cfg["evaluation"]["host"]]:
            raise LaunchError("Docker host IPv4 differs from the explicitly pinned evaluation.host; relay was not started")

    def verify_relay_endpoint(self, state: dict, cfg: dict, *, existing_container: bool = False) -> None:
        # Authenticate the actual coordinator host key and restricted run key
        # from the final firewall namespace. A nonexistent evaluation status
        # request cannot submit/resume work; its precise rejection proves that
        # the forced protocol endpoint was reached. No provider credential is
        # mounted into either of these short, trusted probes.
        ev = cfg["evaluation"]
        absent_id = hashlib.sha256((state["run_id"] + ":relay-probe").encode()).hexdigest()[:32]
        script = f'''import json,subprocess,sys
sys.path.insert(0,"/opt/autoresearch")
from evaluation_access.client import ssh_command
command=ssh_command({(ev["user"] + "@" + ev["host"])!r},"/run/secrets/evaluation_key",
                    port={ev["port"]},known_hosts="/run/secrets/known_hosts")
result=subprocess.run(command,input=json.dumps({{"method":"status","evaluation_id":{absent_id!r}}}),
                      text=True,capture_output=True,timeout=20)
expected={{"ok":False,"error":"Evaluation not found","error_type":"RequestError"}}
assert result.returncode==1 and json.loads(result.stdout)==expected,"restricted evaluator SSH probe failed"
print("ISOLATED_RELAY_READY")
'''
        if existing_container:
            result = self.docker("exec", state["container"], "python3", "-I", "-c", script)
        else:
            result = self.docker(*self.relay_probe_command(state, cfg, script, protected=True))
        if result.stdout.strip() != "ISOLATED_RELAY_READY":
            raise LaunchError("restricted evaluator relay was not verified; agent was not started")

    def auth(self, name: str) -> dict:
        state = self.state(name)
        cfg = json.loads((self.path(name) / "control/config.json").read_text())
        if cfg["auth"]["kind"] != "interactive" or state["status"] not in {"created", "stopped"}:
            raise LaunchError("auth requires interactive authentication and a created/stopped run")
        self.preflight(cfg["image"])
        if not sys.stdin.isatty():
            raise LaunchError("run auth in an interactive terminal; login occurs inside this run's private home")
        for directory in (".codex", ".claude"):
            private_mkdir(self.path(name) / "home", directory, uid=cfg["resources"]["uid"], gid=cfg["resources"]["gid"])
        try:
            self.remove_containers(state)
            self.start_network(state, cfg)
            command = self.agent_command(name, state, cfg)
            command.remove("--detach")
            command[1:1] = ["--interactive", "--tty", "--rm"]
            point = command.index("--entrypoint")
            command[point + 1] = "codex" if cfg["provider"] == "codex" else "claude"
            command[-1:] = (["login", "--device-auth"] if cfg["provider"] == "codex" else ["auth", "login"])
            if subprocess.run(["docker", *command]).returncode:
                raise LaunchError("per-run interactive login did not complete")
            state["authenticated_at"] = now()
            self.save(name, state)
            return self.public_status(name)
        finally:
            cleanup_errors = []
            try:
                self.remove_containers(state)
            except Exception:
                cleanup_errors.append("Docker container removal was not acknowledged")
            try:
                self.stop_relay(state)
            except Exception:
                cleanup_errors.append("SSH relay removal was not acknowledged")
            if cleanup_errors:
                state["status"] = "stopped_unconfirmed"
                self.save(name, state)
                raise LaunchError("; ".join(cleanup_errors) + "; retry stop before continuing")

    def remove_containers(self, state: dict) -> None:
        for key in ("container", "network_container"):
            result = self.docker("inspect", state[key], check=False)
            if result.returncode:
                if self.confirmed_absent(result, state[key]):
                    continue
                raise LaunchError("Docker did not confirm container absence; teardown remains unconfirmed")
            spec = json.loads(result.stdout)[0]
            if spec.get("Config", {}).get("Labels", {}).get("ar.run") != state["run_id"]:
                raise LaunchError("refusing to remove a container without this run's ownership label")
            if key == "container":
                # Stop first so the final output is stable, then retain a
                # redacted host-only copy before Docker deletes its logs.
                from isolated_runs.monitor import archive_container_log
                if spec.get("State", {}).get("Running") is True:
                    self.docker("stop", "--time", "5", state[key])
                archive_container_log(self.path(state["name"]), state, docker=self.docker)
            self.docker("rm", "--force", state[key])

    @staticmethod
    def confirmed_absent(result: subprocess.CompletedProcess, name: str) -> bool:
        return result.returncode != 0 and result.stderr.strip() in {
            f"Error: No such object: {name}", f"Error: No such container: {name}",
            f"Error response from daemon: No such container: {name}",
            f"error: no such object: {name}", f"error: no such container: {name}",
            f"error response from daemon: no such container: {name}"}

    def stop(self, name: str) -> dict:
        state = self.state(name)
        if state["status"] in {"finished", "finishing"}:
            return self.public_status(name)
        # Persist intent before any teardown. Recovery must remain disabled
        # even if this process dies halfway through an operator stop.
        state["status"] = "stopping"
        self.save(name, state)
        # Pause before terminating the submitting process. Server cancellation
        # is owner-scoped and only queued tasks are removed; running xTB work
        # may finish and is recovered on resume.
        errors = []
        try:
            self.admin(name, "set-state", "--state", "paused")
        except Exception:
            errors.append("gateway pause was not acknowledged")
        try:
            self.remove_containers(state)
        except Exception:
            errors.append("Docker container removal was not acknowledged")
        try:
            self.stop_relay(state)
        except Exception:
            errors.append("SSH relay removal was not acknowledged")
        state["status"] = "stopped_unconfirmed" if errors else "stopped"
        self.save(name, state)
        if errors:
            raise LaunchError("; ".join(errors) + "; retry stop before resume/export/finish")
        return self.public_status(name)

    def verification_permit(self, name: str) -> dict:
        """Pause gateway writes before authorizing exactly one verifier session."""
        state = self.state(name)
        if state.get("purpose") != "verification" or state["status"] != "running":
            raise LaunchError("verification-permit requires a running verification-purpose container")
        marker = private_read(self.path(name) / "workspace", "verification/ready.json")
        if marker is None:
            raise LaunchError("fresh train/valid anchors are not ready for verification")
        ready = json.loads(marker)
        expected_hash = digest(self.path(name) / "control/starting-algo.py")
        if ready.get("run_id") != state["run_id"] or ready.get("algo_sha256") != expected_hash:
            raise LaunchError("verification-ready marker does not identify this approved run")
        ids = ready.get("anchor_evaluation_ids")
        if not isinstance(ids, dict) or set(ids) != {"train", "valid"} or len(set(ids.values())) != 2:
            raise LaunchError("verification-ready marker requires distinct train/valid anchor IDs")
        validity = ready.get("anchor_validity")
        if (not isinstance(validity, dict) or set(validity) != {"train", "valid"}
                or any(flag not in (0, 1) for flag in validity.values())):
            raise LaunchError("verification-ready marker must preserve both scientific validity flags")
        observations = {}
        runtime = json.loads((self.path(name) / "control/runtime.json").read_text())
        expected_release_id = runtime.get("expected_release_id")
        if not expected_release_id:
            raise LaunchError("run lacks an approved dataset release identity; create a fresh run")
        for split in ("train", "valid"):
            raw = private_read(self.path(name) / "workspace", f"anchors/{split}.json")
            if raw is None:
                raise LaunchError("verification requires both complete starting observation files")
            observation = json.loads(raw)
            if (observation.get("run_id") != state["run_id"] or observation.get("status") != "complete"
                    or observation.get("candidate_commit") != state["initial_commit"]
                    or observation.get("split") != split or observation.get("evaluation_id") != ids[split]
                    or observation.get("source_sha256") != expected_hash or observation.get("program_id") != expected_hash
                    or not observation.get("timestamp") or observation.get("release_id") != expected_release_id
                    or observation.get("internal_error_count") != 0 or observation.get("is_valid") != validity[split]):
                raise LaunchError("starting observation is incomplete, has internal errors, or has mismatched provenance")
            observations[split] = observation
        if observations["train"]["release_id"] != observations["valid"]["release_id"]:
            raise LaunchError("starting observations use different releases")
        def check_owned_anchors(remote):
            jobs = remote.get("evaluations", [])
            if len(jobs) != 2:
                raise LaunchError("verification requires exactly the two fresh starting evaluations")
            for job in jobs:
                if (job.get("run_id") != state["run_id"] or job.get("status") != "complete"
                        or job.get("candidate_commit") != state["initial_commit"]
                        or job.get("source_sha256") != expected_hash
                        or job.get("evaluation_id") != ids.get(job.get("split"))
                        or job.get("release_id") != observations["train"]["release_id"]):
                    raise LaunchError("gateway ledger does not match the two completed approved anchors")
        check_owned_anchors(self.admin(name, "status"))
        self.admin(name, "set-state", "--state", "paused")
        paused = self.admin(name, "status")
        if paused.get("state") != "paused":
            raise LaunchError("gateway did not confirm read-only verification access")
        check_owned_anchors(paused)
        permit = {"run_id": state["run_id"], "algo_sha256": expected_hash, "anchor_evaluation_ids": ids,
                  "anchor_validity": validity, "gateway_state": "paused", "issued_at": now()}
        permit_path = self.path(name) / "control/verification-permit/permit.json"
        write_json(permit_path, permit)
        permit_path.chmod(0o644)
        return {"run_id": state["run_id"], "verification_permitted": True, "gateway_state": "paused",
                "anchor_validity": validity}

    def public_status(self, name: str) -> dict:
        state = self.state(name)
        result = {k: state[k] for k in ("name", "run_id", "status", "created_at", "updated_at", "key_revoked")}
        label = state.get("display_name")
        result["display_name"] = label if isinstance(label, str) and NAME.fullmatch(label) else state["name"]
        result["purpose"] = state.get("purpose", "research")
        cfg = json.loads((self.path(name) / "control/config.json").read_text())
        if "relay" in cfg["evaluation"]:
            try:
                result["relay"] = self.relay(state, cfg).status()
            except (RelayError, OSError, ValueError, KeyError) as exc:
                result["relay"] = {"state": "unavailable", "detail": str(exc)}
        if state["status"] in {"running", "starting", "failed", "stopping", "stopped", "stopped_unconfirmed", "finishing"}:
            try:
                inspection = self.docker("inspect", "--format", "{{json .State}}", state["container"], check=False)
                result["container_state"] = (json.loads(inspection.stdout) if inspection.returncode == 0 else
                    "absent" if self.confirmed_absent(inspection, state["container"]) else "unknown: Docker unavailable")
            except LaunchError:
                result["container_state"] = "unknown: Docker unavailable"
        return result

    def status(self, name: str) -> dict:
        result = self.public_status(name)
        try:
            result["evaluation"] = self.admin(name, "status")
        except LaunchError as exc:
            result["evaluation"] = {"status": "unavailable", "detail": str(exc)}
        ready = private_read(self.path(name) / "workspace", ".run-ready.json")
        result["research_gate"] = json.loads(ready) if ready else "pending fresh anchors or recovery"
        return result

    def export(self, name: str) -> dict:
        state, path = self.state(name), self.path(name)
        # Reuse the vetted final archive: secrets have been removed after
        # finish, so a fresh export could no longer detect disguised key copies.
        if state["status"] == "finished":
            return state["archive"]
        if state["status"] not in {"stopped", "finished", "created", "creation_failed"}:
            raise LaunchError("stop the run before exporting a consistent private archive")
        secret_values = []
        for relative in ("provider", "evaluation_key"):
            data = private_read(path / "secrets", relative)
            if data is not None:
                data = data.strip()
                secret_values.extend([data, *[line for line in data.splitlines() if len(line) > 24]])
        def collect_strings(value):
            if isinstance(value, str) and len(value) >= 8:
                secret_values.append(value.encode())
            elif isinstance(value, dict):
                for item in value.values():
                    collect_strings(item)
            elif isinstance(value, list):
                for item in value:
                    collect_strings(item)
        for relative in (".codex/auth.json", ".claude/.credentials.json", ".claude.json"):
            auth_data = private_read(path / "home", relative)
            if auth_data is not None:
                try:
                    collect_strings(json.loads(auth_data))
                except (ValueError, UnicodeError):
                    secret_values.append(auth_data.strip())
        # These exact diagnostic paths are exceptions to the normal rejection
        # of hidden/state directories. Enumerate only flat regular log files
        # through directory descriptors, never by following agent symlinks.
        from isolated_runs.monitor import LOG_NAME, redact
        diagnostics = {PurePosixPath("workspace/.ralph/ralph-loop.state.json"),
                       PurePosixPath("workspace/.ralph/ralph-history.json")}
        for directory, pattern in (("workspace/.ralph/logs", LOG_NAME), ("workspace/logs", LOG_NAME),
                                   ("operator-logs", re.compile(r"docker-[0-9]+\.log\Z"))):
            try:
                with private_parent(path, directory + "/entry") as (fd, _):
                    with os.scandir(fd) as entries:
                        for item in entries:
                            if pattern.fullmatch(item.name) and item.is_file(follow_symlinks=False):
                                diagnostics.add(PurePosixPath(directory) / item.name)
            except (FileNotFoundError, NotADirectoryError):
                pass
            except OSError as exc:
                if exc.errno != errno.ELOOP:
                    raise
        archive = path / "exports" / f'{state["run_id"]}-{time.time_ns()}.tar.gz'
        omitted, included_diagnostics, redacted_diagnostics = [], [], []
        with archive.open("xb") as handle:
            os.chmod(archive, 0o600)
            with tarfile.open(fileobj=handle, mode="w:gz", dereference=False) as tar:
                for area in ("workspace", "memory"):
                    for item in sorted((path / area).rglob("*")):
                        rel = item.relative_to(path / area)
                        if any(p.startswith(".") or p in DENIED_PARTS for p in rel.parts):
                            continue
                        data = private_read(path / area, str(rel))
                        if data is None:
                            continue
                        if item.name in {"auth.json", "credentials.json"} or item.suffix in {".key", ".pem"} or any(
                                value and value in data for value in secret_values) or b"-----BEGIN OPENSSH PRIVATE KEY-----" in data:
                            omitted.append(str(PurePosixPath(area) / rel))
                            continue
                        info = tarfile.TarInfo(str(PurePosixPath(area) / rel))
                        info.size, info.mode = len(data), 0o600
                        tar.addfile(info, io.BytesIO(data))
                diagnostic_secrets = sorted({value.decode("utf-8", "replace") for value in secret_values if value},
                                            key=len, reverse=True)
                for relative in sorted(diagnostics):
                    original = private_read(path, str(relative))
                    if original is None:
                        continue
                    data = redact(original.decode("utf-8", "replace"), diagnostic_secrets).encode()
                    if any(value and value in data for value in secret_values):
                        omitted.append(str(relative))
                        continue
                    info = tarfile.TarInfo(str(relative))
                    info.size, info.mode = len(data), 0o600
                    tar.addfile(info, io.BytesIO(data))
                    included_diagnostics.append(str(relative))
                    if data != original:
                        redacted_diagnostics.append(str(relative))
                manifest = {"schema": SCHEMA, "run_id": state["run_id"], "initial_commit": state.get("initial_commit"),
                            "created_at": state["created_at"], "exported_at": now(), "omitted_credentials": omitted,
                            "included_diagnostics": included_diagnostics, "redacted_diagnostics": redacted_diagnostics,
                            "evaluation_status": self.admin(name, "status"),
                            "snapshot_sha256": json.loads((path / "control/config.json").read_text())["snapshot"]["files"]}
                data = json.dumps(manifest, indent=2).encode()
                info = tarfile.TarInfo("manifest.json")
                info.size, info.mode = len(data), 0o600
                tar.addfile(info, io.BytesIO(data))
        return {"archive": str(archive), "sha256": digest(archive), "omitted_credentials": omitted}

    def finish(self, name: str) -> dict:
        state = self.state(name)
        if state["status"] == "finished":
            return self.public_status(name)
        if state["status"] != "finishing":
            self.stop(name)
            result = self.export(name)
            # Persist the cleanup stage before revocation. A crash here retries
            # revocation and private cleanup; it can never silently re-enable a
            # revoked key by falling through stop/export on the next attempt.
            state = self.state(name)
            state["status"] = "finishing"
            state["archive"] = result
            self.save(name, state)
        self.admin(name, "set-state", "--state", "revoked")
        state["key_revoked"] = True
        self.save(name, state)
        for item in (self.path(name) / "secrets").iterdir():
            private_unlink(self.path(name) / "secrets", item.name)
        for relative in (".codex/auth.json", ".claude/.credentials.json"):
            private_unlink(self.path(name) / "home", relative)
        self.docker("network", "rm", state["network"], check=False)
        state["status"] = "finished"
        self.save(name, state)
        return {**self.public_status(name), **state["archive"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="private host directory containing isolated runs")
    parser.add_argument("--provider", choices=("codex", "claude"), help="wrapper provider consistency check")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("create").add_argument("--config", type=Path, required=True)
    for action in ("auth", "start", "status", "stop", "resume", "finish", "export", "verification-permit", "ensure-relay"):
        sub.add_parser(action).add_argument("name")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config) if args.action == "create" else None
        name = cfg["name"] if cfg else args.name
        launcher = Launcher(args.root)
        if args.provider:
            actual = cfg["provider"] if cfg else json.loads((launcher.path(name) / "control/config.json").read_text())["provider"]
            if actual != args.provider:
                raise LaunchError("launcher wrapper provider differs from the explicit run configuration")
        with launcher.lock(name):
            if args.action == "create":
                result = launcher.create(cfg)
            elif args.action == "resume":
                result = launcher.start(name, resume=True)
            else:
                result = getattr(launcher, args.action.replace("-", "_"))(name)
        print(json.dumps(result, indent=2))
        return 0
    except (RelayError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"isolated-run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
