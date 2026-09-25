"""Read-only operator dashboard for a private isolated-run root.

Run with ``python -m isolated_runs.monitor --root /private/runs --port 8765``.
The agent firewall does not grant access to this loopback-only host service.
No provider commands, Git commands, or lifecycle mutations are executed.
Optional gateway ledger polling runs outside HTTP handlers and invokes only
the private configured admin argv with the fixed read-only status operation.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import stat
import subprocess
import tarfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

from isolated_runs.monitor_progress import (MAX_TABLE_BYTES, TABLE_NAMES, TABLE_PATHS,
    build_progress, parse_table, summary as table_summary)

NAME = re.compile(r"[a-z][a-z0-9-]{0,39}\Z")
HEX = re.compile(r"[a-f0-9]{32,64}\Z")
TOKEN = re.compile(r"[A-Za-z0-9._:/-]{1,128}\Z")
LOG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\.log(?:\.[0-9]{1,3})?\Z")
MAX_JSON = 2 * 1024 * 1024
MAX_TAIL = 256 * 1024
MAX_ARCHIVE = 32 * 1024 * 1024
MAX_FILES = 128
MAX_EXPORT_BYTES = 1024 * 1024 * 1024
MAX_EXPORT_MEMBERS = 8192
STATUSES = {"creating", "created", "creation_failed", "starting", "running", "failed", "stopped",
            "stopped_unconfirmed", "finishing", "finished", "ready", "verification_waiting_for_permit",
            "verification_complete", "complete", "pending", "queued", "cancelled", "cancelling",
            "enabled", "paused", "revoked", "exited", "dead", "restarting", "removing"}
FIXED_LOGS = {
    "provider": "workspace/verification/provider.log",
    "verification-summary": "workspace/verification/summary.md",
    "ralph-state": "workspace/.ralph/ralph-loop.state.json",
    "ralph-history": "workspace/.ralph/ralph-history.json",
    "run-log": "workspace/run.log",
    "full-log": "workspace/full_log.md",
    "backlog": "workspace/backlog.md",
    "results": "workspace/results.tsv",
    "generalizable": "workspace/generalizable.tsv",
    "non_generalizable": "workspace/non_generalizable.tsv",
    "validation-debug": "workspace/validate_debug.log",
}
LOG_DIRS = {"ralph": "workspace/.ralph/logs", "debug": "workspace/logs", "archive": "operator-logs"}
EXPORT_METADATA = {"workspace/anchors/train.json", "workspace/anchors/valid.json",
                   "workspace/verification/invocation.json", "workspace/verification/complete.json"}
SEALED_STATES = {"finishing", "finished"}


class MonitorError(RuntimeError):
    pass


def utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def token(value, default="unknown"):
    return value if isinstance(value, str) and TOKEN.fullmatch(value) else default


def status(value):
    return value if isinstance(value, str) and value in STATUSES else "unknown"


def timestamp(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 32503680000:
        return utc(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9T:+.Z-]{10,40}", value):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass
    return None


def hexadecimal(value):
    return value if isinstance(value, str) and HEX.fullmatch(value) else None


def open_absolute_directory(path: Path) -> int:
    """Pin the directory and reject symlinks in every absolute component."""
    path = path.expanduser().absolute()
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if part in {".", ".."}:
                raise MonitorError("invalid root path")
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise MonitorError("run root must be owned by this operator and private (chmod 700)")
        return fd
    except BaseException:
        os.close(fd)
        raise


class Files:
    """Reads relative to a pinned directory; no path component follows a link."""
    def __init__(self, fd: int):
        self.fd = fd

    @contextmanager
    def open(self, relative: str, *, directory=False):
        parts = PurePosixPath(relative).parts
        if (not parts or relative.startswith("/") or str(PurePosixPath(relative)) != relative
                or any(part in {"..", "."} for part in parts)):
            raise MonitorError("invalid relative path")
        parent = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if directory:
                flags |= os.O_DIRECTORY
            fd = os.open(parts[-1], flags, dir_fd=parent)
            try:
                info = os.fstat(fd)
                if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
                    raise MonitorError("not a regular file/directory")
                yield fd, info
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    def read(self, relative: str, *, tail=False, limit=MAX_JSON):
        try:
            with self.open(relative) as (fd, info):
                if info.st_size > limit and not tail:
                    return None
                if tail:
                    os.lseek(fd, max(0, info.st_size - limit), os.SEEK_SET)
                chunks = []
                remaining = limit
                while remaining:
                    chunk = os.read(fd, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b"".join(chunks), info.st_mtime, info.st_size > limit
        except (OSError, MonitorError):
            return None

    def json(self, relative: str):
        result = self.read(relative)
        try:
            value = json.loads(result[0]) if result else None
            return value if isinstance(value, dict) else {}
        except (ValueError, UnicodeError, RecursionError):
            return {}

    def table(self, name):
        """Read a complete bounded TSV snapshot, never a headerless tail."""
        try:
            with self.open(TABLE_PATHS[name]) as (fd, before):
                metadata = {"modified": utc(before.st_mtime), "size": before.st_size}
                if before.st_size > MAX_TABLE_BYTES:
                    return parse_table(None, availability="oversized", **metadata)
                chunks, remaining = [], MAX_TABLE_BYTES + 1
                while remaining:
                    block = os.read(fd, min(65536, remaining))
                    if not block:
                        break
                    chunks.append(block)
                    remaining -= len(block)
                after = os.fstat(fd)
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    return parse_table(None, availability="changing; retry next refresh", **metadata)
                return parse_table(b"".join(chunks), **metadata)
        except FileNotFoundError:
            return parse_table(None, availability="missing")
        except (OSError, MonitorError):
            return parse_table(None, availability="unavailable or unsafe")

    def names(self, relative: str):
        try:
            with self.open(relative, directory=True) as (fd, _):
                # scandir streams instead of allocating an unbounded listing.
                with os.scandir(fd) as entries:
                    result = []
                    for entry in entries:
                        if len(result) >= MAX_FILES:
                            break
                        if LOG_NAME.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):
                            result.append(entry.name)
                    return sorted(result)
        except (OSError, MonitorError):
            return []


class ExportFiles(Files):
    """An in-memory, already vetted export view; never falls back to disk."""
    def __init__(self, members=None, modified=0):
        self.members = members or {}
        self.modified = modified

    def read(self, relative, *, tail=False, limit=MAX_JSON):
        item = self.members.get(relative)
        if item is None:
            return None
        data, size = item
        if not tail and (size > limit or size > len(data)):
            return None
        return data[-limit:] if tail else data, self.modified, size > limit

    def names(self, relative):
        prefix = relative + "/"
        return sorted(path[len(prefix):] for path in self.members if path.startswith(prefix)
                      and "/" not in path[len(prefix):] and LOG_NAME.fullmatch(path[len(prefix):]))[:MAX_FILES]

    def table(self, name):
        item = self.members.get(TABLE_PATHS[name])
        if item is None:
            return parse_table(None, availability="not present in verified export")
        data, size = item
        metadata = {"modified": utc(self.modified), "size": size}
        if size > MAX_TABLE_BYTES or len(data) < size:
            return parse_table(None, availability="oversized", **metadata)
        return parse_table(data, **metadata)


class ExportBudget:
    """Bound gzip expansion and parser time, including skipped tar members."""
    def __init__(self, stream, deadline):
        self.stream, self.deadline, self.used = stream, deadline, 0

    def read(self, size=-1):
        if time.monotonic() > self.deadline:
            raise MonitorError("export read time limit")
        amount = MAX_EXPORT_BYTES - self.used + 1
        data = self.stream.read(min(amount, size) if size >= 0 else amount)
        self.used += len(data)
        if self.used > MAX_EXPORT_BYTES:
            raise MonitorError("export expansion limit")
        return data


class ExportTarInfo(tarfile.TarInfo):
    extended_bytes = 0

    @classmethod
    def frombuf(cls, buf, encoding, errors):
        member = super().frombuf(buf, encoding, errors)
        if member.type in {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}:
            cls.extended_bytes += member.size
            if member.size > 65536 or cls.extended_bytes > 1024 * 1024:
                raise MonitorError("export extended header limit")
        return member


def exported_path_allowed(name):
    if name in FIXED_LOGS.values() or name in EXPORT_METADATA:
        return True
    return any(name.startswith(directory + "/") and "/" not in name[len(directory) + 1:]
               and LOG_NAME.fullmatch(name[len(directory) + 1:]) for directory in LOG_DIRS.values())


def read_export(fd, *, expected_sha256, run_id, modified):
    """Verify the host-pinned archive; read bounded tails, never extract files."""
    deadline = time.monotonic() + 10
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    consumed = 0
    while data := os.read(fd, 1024 * 1024):
        consumed += len(data)
        if consumed > MAX_EXPORT_BYTES or time.monotonic() > deadline:
            raise MonitorError("export verification limit")
        digest.update(data)
    if digest.hexdigest() != expected_sha256:
        raise MonitorError("export digest mismatch")
    os.lseek(fd, 0, os.SEEK_SET)
    members, seen, manifest = {}, set(), None
    with os.fdopen(os.dup(fd), "rb") as source:
        with gzip.GzipFile(fileobj=source, mode="rb") as expanded:
            info_type = type("BoundedExportInfo", (ExportTarInfo,), {"extended_bytes": 0})
            with tarfile.open(fileobj=ExportBudget(expanded, deadline), mode="r|", tarinfo=info_type) as archive:
                for count, member in enumerate(archive):
                    if count >= MAX_EXPORT_MEMBERS or not 0 <= member.size <= MAX_EXPORT_BYTES:
                        raise MonitorError("export member limit")
                    if member.name != "manifest.json" and not exported_path_allowed(member.name):
                        continue
                    if (member.name in seen or not member.isreg() or member.issparse()
                            or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}):
                        raise MonitorError("unsafe or duplicate export member")
                    seen.add(member.name)
                    if len(seen) > MAX_FILES + len(EXPORT_METADATA) + 1:
                        raise MonitorError("export log count limit")
                    stream = archive.extractfile(member)
                    if member.name == "manifest.json":
                        if member.size > MAX_JSON:
                            raise MonitorError("export manifest limit")
                        manifest = json.loads(stream.read(MAX_JSON + 1))
                    else:
                        tail = bytearray()
                        limit = (MAX_TABLE_BYTES if member.name in TABLE_PATHS.values()
                                 else MAX_JSON if member.name in EXPORT_METADATA else MAX_TAIL)
                        while block := stream.read(65536):
                            tail.extend(block)
                            if len(tail) > limit:
                                del tail[:-limit]
                        members[member.name] = (bytes(tail), member.size)
    if (not isinstance(manifest, dict) or manifest.get("schema") != 1 or manifest.get("run_id") != run_id
            or not isinstance(manifest.get("included_diagnostics"), list)):
        raise MonitorError("export lacks the owned diagnostic manifest")
    vetted = {name for name in manifest["included_diagnostics"] if isinstance(name, str)}
    for name in members:
        if (name.startswith(("workspace/.ralph/", "workspace/logs/", "operator-logs/")) and name not in vetted):
            raise MonitorError("diagnostic absent from vetted export manifest")
    return ExportFiles(members, modified)


def secrets_for(files: Files):
    """Read only known credentials for redaction; never return their contents."""
    values = set()
    for path in ("secrets/provider", "secrets/evaluation_key"):
        found = files.read(path, limit=MAX_JSON)
        if found:
            content = found[0].decode("utf-8", "replace").strip()
            if content:
                values.add(content)
                values.update(line for line in content.splitlines() if len(line) > 24)

    def collect(value):
        if isinstance(value, str) and len(value) >= 8:
            values.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    for path in ("home/.codex/auth.json", "home/.claude/.credentials.json", "home/.claude.json"):
        collect(files.json(path))
    return sorted(values, key=len, reverse=True)


def redact(text: str, secrets=()) -> str:
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
        escaped = json.dumps(secret)[1:-1]
        if escaped != secret:
            text = text.replace(escaped, "[REDACTED]")
    text = re.sub(r"-----BEGIN [^-\n]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-\n]*PRIVATE KEY-----|$)",
                  "[REDACTED PRIVATE KEY]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{10,}", "[REDACTED KEY]", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", text)
    text = re.sub(r'(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|authorization)["\s:=]+)[^\s,"}]+',
                  r"\1[REDACTED]", text)
    # Remove OSC title/hyperlink payloads as well as ordinary ANSI CSI codes.
    text = re.sub(r"\x1b\][\s\S]*?(?:\x07|\x1b\\|$)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(c for c in text if c in "\n\t" or ord(c) >= 32)


def bounded_command(argv: list[str], *, limit=MAX_TAIL, timeout=5) -> subprocess.CompletedProcess:
    """No shell, finite wall time and memory, including pathological log lines."""
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MonitorError("local Docker request timed out")
                for key, _ in selector.select(min(remaining, .25)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        break
                    output.extend(data)
                    if len(output) > limit:
                        raise MonitorError("local Docker output exceeded the bounded view; use a persisted log")
            code = process.wait(timeout=max(.1, deadline - time.monotonic()))
        return subprocess.CompletedProcess(argv, code, output.decode("utf-8", "replace"), "")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait(timeout=2)
        process.stdout.close()


class Docker:
    def __init__(self):
        self.endpoint = None

    def __call__(self, *args, check=False):
        if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
            raise MonitorError("remote Docker overrides are forbidden")
        if not self.endpoint:
            result = bounded_command(["docker", "context", "inspect", "--format",
                                      '{{(index .Endpoints "docker").Host}}'], limit=8192)
            endpoint = result.stdout.strip()
            if result.returncode or not endpoint.startswith("unix:///") or len(endpoint) > 4096 or "\n" in endpoint:
                raise MonitorError("a local Unix Docker socket is required")
            self.endpoint = endpoint
        return bounded_command(["docker", "--host", self.endpoint, *args])


def archive_container_log(run_path: Path, state: dict, *, docker) -> str:
    """Lifecycle hook after stop/before rm; persist available Docker log tail.

    Caller must verify container ownership first. Only the agent stream is
    archived. Docker rotation may already have discarded older output; this
    archive does not claim to be a complete transcript.
    """
    run_id = state.get("run_id", "")
    if (not isinstance(run_id, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}-[a-f0-9]{16}", run_id)
            or state.get("container") != "ar-agent-" + run_id):
        raise MonitorError("invalid archive run identity")
    fd = open_absolute_directory(run_path)
    try:
        files = Files(fd)
        result = docker("logs", "--timestamps", "--tail", "20000", state["container"], check=False)
        if result.returncode:
            raise MonitorError("Docker log snapshot failed; container retained for retry")
        content = (result.stdout + result.stderr).encode("utf-8", "replace")[-MAX_ARCHIVE:]
        content = redact(content.decode("utf-8", "replace"), secrets_for(files)).encode()
        content = b"Available Docker tail at stop (up to 20000 lines / 32 MiB; earlier Docker rotation may be absent).\n" + content
        try:
            os.mkdir("operator-logs", mode=0o700, dir_fd=fd)
        except FileExistsError:
            pass
        with files.open("operator-logs", directory=True) as (directory, info):
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise MonitorError("operator log directory must be private")
            name = f"docker-{time.time_ns()}.log"
            output_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600, dir_fd=directory)
            with os.fdopen(output_fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        return name
    finally:
        os.close(fd)


class Monitor:
    def __init__(self, root: Path, *, docker=None, gateway_status=False, admin_runner=bounded_command):
        self.fd = open_absolute_directory(root)
        self.root = root.expanduser().absolute()
        self.docker = docker if docker is not None else Docker()
        self.gateway_status = gateway_status
        self.admin_runner = admin_runner
        self.gateway_cache = {}
        self.export_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.stop_poll = threading.Event()
        self.poll_thread = None
        if gateway_status:
            self.poll_thread = threading.Thread(target=self.poll_gateways, daemon=True)
            self.poll_thread.start()

    def close(self):
        self.stop_poll.set()
        if self.poll_thread:
            self.poll_thread.join(timeout=12)
        os.close(self.fd)

    def poll_gateways(self):
        while not self.stop_poll.is_set():
            for name in self.names():
                if self.stop_poll.is_set():
                    break
                self.poll_gateway(name)
            self.stop_poll.wait(30)

    def poll_gateway(self, name):
        """Never receives an executable, action, or run ID from a web request."""
        observed = utc(time.time())
        try:
            with self.run(name) as (files, saved):
                if saved.get("status") in SEALED_STATES:
                    return  # Finished runs use their vetted historical export.
                command = files.json("control/config.json").get("evaluation", {}).get("admin_command")
                if (not isinstance(command, list) or not 1 <= len(command) <= 64
                        or not all(isinstance(arg, str) and 0 < len(arg) <= 4096 and "\0" not in arg for arg in command)):
                    raise MonitorError("no configured admin argv")
                response = self.admin_runner([*command, "status", "--run-id", saved["run_id"]], limit=MAX_JSON, timeout=10)
                raw = json.loads(response.stdout)
                if (response.returncode or not isinstance(raw, dict) or raw.get("ok") is False
                        or raw.get("run_id") != saved["run_id"] or not isinstance(raw.get("evaluations"), list)):
                    raise MonitorError("no owned gateway acknowledgement")
                value = {"available": True, "observed_at": observed, "state": status(raw.get("state")),
                         "evaluations": [self.evaluation(item, saved["run_id"])
                                         for item in raw["evaluations"][-MAX_FILES:]]}
                value = self.redacted_value(value, secrets_for(files))
        except (OSError, MonitorError, ValueError, TypeError, AttributeError, RecursionError):
            # Do not expose admin args, error messages, remote stderr, or config.
            value = {"available": False, "attempted_at": observed}
        with self.cache_lock:
            if not value["available"] and name in self.gateway_cache:
                value = {**self.gateway_cache[name], **value}
            self.gateway_cache[name] = value

    @staticmethod
    def redacted_value(value, secrets):
        if isinstance(value, str):
            return redact(value, secrets)
        if isinstance(value, dict):
            return {key: Monitor.redacted_value(item, secrets) for key, item in value.items()}
        if isinstance(value, list):
            return [Monitor.redacted_value(item, secrets) for item in value]
        return value

    def exported_files(self, files, saved):
        info = saved.get("archive")
        if not isinstance(info, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(info.get("sha256", ""))):
            raise MonitorError("no verified export")
        filename = Path(str(info.get("archive", ""))).name
        if (not re.fullmatch(re.escape(saved["run_id"]) + r"-[0-9]{1,24}\.tar\.gz", filename)
                or info.get("archive") != str(self.root / saved["name"] / "exports" / filename)):
            raise MonitorError("export must be pinned inside this run")
        with files.open("exports/" + filename) as (fd, before):
            if before.st_uid != os.getuid() or before.st_mode & 0o077 or before.st_size > MAX_EXPORT_BYTES:
                raise MonitorError("export must be private and bounded")
            key = (saved["run_id"], info["sha256"], before.st_ino, before.st_size, before.st_mtime_ns)
            if key in self.export_cache:
                self.export_cache.move_to_end(key)
                return self.export_cache[key]
            result = read_export(fd, expected_sha256=info["sha256"], run_id=saved["run_id"], modified=before.st_mtime)
            after = os.fstat(fd)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise MonitorError("export changed during verification")
            self.export_cache[key] = result
            while len(self.export_cache) > 4:
                self.export_cache.popitem(last=False)
            return result

    @contextmanager
    def run(self, name: str):
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise MonitorError("unknown run")
        try:
            with Files(self.fd).open(name, directory=True) as (fd, info):
                if info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise MonitorError("run directory is not private")
                files = Files(fd)
                saved = files.json("control/state.json")
                run_id = saved.get("run_id")
                if (saved.get("schema") != 1 or saved.get("name") != name
                        or not isinstance(run_id, str)
                        or not re.fullmatch(re.escape(name) + r"-[a-f0-9]{16}", run_id)):
                    raise MonitorError("unknown run")
                yield files, saved
        except OSError as exc:
            raise MonitorError("unknown run") from exc

    def names(self):
        result = []
        with os.scandir(self.fd) as entries:
            for entry in entries:
                if len(result) >= MAX_FILES:
                    break
                if NAME.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False):
                    try:
                        with self.run(entry.name):
                            result.append(entry.name)
                    except MonitorError:
                        pass
        return sorted(result)

    def display_names(self):
        """Labels may change while directory names and run identities stay fixed."""
        labels = {}
        for name in self.names():
            try:
                with self.run(name) as (_, saved):
                    label = saved.get("display_name")
                    labels[name] = label if isinstance(label, str) and NAME.fullmatch(label) else name
            except MonitorError:
                continue  # A stopped run may have been deleted during refresh.
        return labels

    def container(self, saved):
        run_id = saved["run_id"]
        name = "ar-agent-" + run_id
        # No arbitrary container identifier is accepted from the query or state.
        fmt = '{{json .Config.Labels}}\n{{json .State}}\n{{json .Config.Entrypoint}}\n{{json .Config.Cmd}}'
        try:
            reply = self.docker("inspect", "--format", fmt, name, check=False)
            if reply.returncode:
                return {"status": "absent_or_unavailable", "running": False}
            lines = reply.stdout.splitlines()
            labels, state = json.loads(lines[0]), json.loads(lines[1])
            if labels.get("ar.run") != run_id:
                return {"status": "ownership_mismatch", "running": False}
            entrypoint = json.loads(lines[2]) if len(lines) > 2 else []
            command = json.loads(lines[3]) if len(lines) > 3 else []
            login = (isinstance(entrypoint, list) and entrypoint and isinstance(entrypoint[0], str)
                     and isinstance(command, list) and (
                         (Path(entrypoint[0]).name == "codex" and command[:1] == ["login"])
                         or (Path(entrypoint[0]).name == "claude" and command[:2] == ["auth", "login"])))
            return {"status": status(state.get("Status")), "running": state.get("Running") is True,
                    "phase": "sign_in" if login else "run",
                    "exit_code": state.get("ExitCode") if state.get("Running") is not True and type(state.get("ExitCode")) is int else None,
                    "started_at": timestamp(state.get("StartedAt")), "finished_at": timestamp(state.get("FinishedAt")),
                    "owned": True}
        except (OSError, MonitorError, ValueError, IndexError, TypeError, AttributeError):
            return {"status": "unavailable", "running": False}

    @staticmethod
    def logs(files, purpose):
        logs = {}
        for key, path in FIXED_LOGS.items():
            if purpose == "verification" and key.startswith("ralph"):
                continue
            if files.read(path, tail=True, limit=1):
                logs[key] = path
        for prefix, path in LOG_DIRS.items():
            if prefix == "ralph" and purpose == "verification":
                continue
            for name in files.names(path):
                logs[prefix + "/" + name] = path + "/" + name
        return logs

    def detail(self, name):
        with self.run(name) as (files, saved):
            runtime = files.json("control/runtime.json")
            purpose = "verification" if saved.get("purpose") == "verification" else "research"
            sealed = saved.get("status") in SEALED_STATES
            log_access = "Live and persisted run logs"
            if sealed:
                try:
                    files = self.exported_files(files, saved)
                    log_access = "Verified sanitized export; raw workspace logs are not served after finish"
                except (OSError, MonitorError, ValueError, EOFError, tarfile.TarError, RecursionError):
                    files = ExportFiles()
                    log_access = "Sanitized export unavailable or failed verification; raw logs remain inaccessible"
            container = {"status": "finished", "running": False} if sealed else self.container(saved)
            result = {"name": name, "run_id": saved["run_id"], "purpose": purpose,
                      "provider": token(runtime.get("provider")), "model": token(runtime.get("model")),
                      "effort": token(runtime.get("effort")), "lifecycle": status(saved.get("status")),
                      "updated_at": timestamp(saved.get("updated_at")), "container": container,
                      "evaluation_source": "Local saved receipts; not a live gateway poll",
                      "initial_candidate": hexadecimal(saved.get("initial_commit"))}
            label = saved.get("display_name")
            result["display_name"] = label if isinstance(label, str) and NAME.fullmatch(label) else name
            result["log_access"] = log_access
            tables = {key: files.table(key) for key in TABLE_NAMES}
            result["tables"] = {key: table_summary(value) for key, value in tables.items()}
            result["research_progress"] = build_progress(tables, initial_commit=saved.get("initial_commit", ""))
            if sealed:
                result["research_progress"]["source"] = "Verified sanitized export; historical snapshot of this run"
            # Read Git references as data, without loading config or running hooks.
            head = files.read("workspace/.git/HEAD", limit=4096)
            current = head[0].decode("ascii", "replace").strip() if head else ""
            if current.startswith("ref: "):
                reference = current[5:]
                if re.fullmatch(r"refs/heads/[A-Za-z0-9_./-]{1,160}", reference) and ".." not in reference:
                    found = files.read("workspace/.git/" + reference, limit=4096)
                    current = found[0].decode("ascii", "replace").strip() if found else ""
                    if not current:
                        packed = files.read("workspace/.git/packed-refs", limit=MAX_JSON)
                        if packed:
                            for line in packed[0].decode("ascii", "replace").splitlines():
                                parts = line.split()
                                if len(parts) == 2 and parts[1] == reference:
                                    current = parts[0]
            result["current_candidate"] = hexadecimal(current)
            ready = files.json("workspace/.run-ready.json")
            result["gate"] = status(ready.get("status")) if ready.get("run_id") == saved["run_id"] else "pending anchors or recovery"
            anchors, evaluation = {}, []
            for split in ("train", "valid"):
                raw = files.json(f"workspace/anchors/{split}.json")
                item = self.evaluation(raw, saved["run_id"])
                owned = (raw.get("run_id") == saved["run_id"] and raw.get("split") == split
                         and raw.get("candidate_commit") == saved.get("initial_commit"))
                valid = (raw.get("status") == "complete" and owned
                         and raw.get("is_valid") == 1 and raw.get("internal_error_count") == 0
                         and hexadecimal(raw.get("evaluation_id")) and token(raw.get("release_id")) != "unknown")
                anchors[split] = {**item, "ready": bool(valid), "started": bool(raw),
                                  "status_source": "saved anchor result" if raw else "not started"}
                if owned and raw.get("status") == "complete":
                    if type(raw.get("is_valid")) is int and raw["is_valid"] in {0, 1}:
                        anchors[split]["is_valid"] = raw["is_valid"]
                    reason = raw.get("invalid_reason") or raw.get("validity_reason")
                    if raw.get("is_valid") == 0:
                        anchors[split]["invalid_reason"] = token(reason, "unspecified")
                    for field in ("internal_error_count", "num_results", "molecule_count"):
                        if type(raw.get(field)) is int and 0 <= raw[field] <= 10**9:
                            anchors[split][field] = raw[field]
                    energy = raw.get("mean_rel_energy")
                    # Internal-error scores contain a penalty sentinel, not a
                    # measured aggregate energy ratio.
                    if (raw.get("internal_error_count") == 0
                            and type(energy) in {int, float} and math.isfinite(energy)):
                        anchors[split]["mean_rel_energy"] = energy
                if not raw:
                    anchors[split]["status"] = "not_started"
            result["anchors"] = anchors
            result["anchors_ready"] = all(x["ready"] for x in anchors.values()) and (
                files.json("workspace/anchors/train.json").get("release_id") ==
                files.json("workspace/anchors/valid.json").get("release_id"))
            submissions = {}
            for path in ("workspace/.evaluation_state/client.json", "workspace/.evaluation_state/anchors.json"):
                entries = files.json(path).get("evaluations", {})
                if isinstance(entries, dict):
                    # Keep submission dates for older plotted cycles too, even
                    # when they fall outside the bounded live-status list.
                    for entry in entries.values():
                        receipt = entry.get("receipt", {}) if isinstance(entry, dict) else {}
                        if isinstance(receipt, dict) and receipt.get("run_id") == saved["run_id"]:
                            item = self.evaluation(receipt, saved["run_id"])
                            if item.get("evaluation_id") and item.get("submitted_at"):
                                submissions[item["evaluation_id"]] = item
                    for entry in list(entries.values())[-MAX_FILES:]:
                        if isinstance(entry, dict):
                            receipt = entry.get("receipt", {})
                            if isinstance(receipt, dict):
                                identity = entry.get("evaluation_id") or receipt.get("evaluation_id")
                                if (hexadecimal(identity) and receipt.get("evaluation_id") in {None, identity}
                                        and receipt.get("run_id") == saved["run_id"]):
                                    item = self.evaluation({**receipt, "evaluation_id": identity}, saved["run_id"])
                                    # A submit receipt is historical, not a live
                                    # queue observation. Keep its original state
                                    # for diagnostics without presenting it as active.
                                    evaluation.append({**item, "submission_status": item["status"],
                                                       "status": "unknown"})
                                    split = item["split"]
                                    # The runner persists submit receipts here before
                                    # anchors/<split>.json exists. Only this starting
                                    # candidate and the explicit anchor request may
                                    # supply a missing anchor card.
                                    if (path.endswith("/anchors.json") and split in anchors
                                            and entry.get("request_id") == "anchor-" + split
                                            and item["candidate_commit"] == result["initial_candidate"]
                                            and not anchors[split].get("evaluation_id")):
                                        anchors[split] = {**item, "ready": False, "started": True,
                                                          "status_source": "saved submission receipt"}
                                elif path.endswith("/anchors.json") and not receipt and not identity:
                                    for split in anchors:
                                        if entry.get("request_id") == "anchor-" + split and not anchors[split]["started"]:
                                            anchors[split].update(status="submitting", started=True,
                                                status_source="saved request; submission not acknowledged")
            result["evaluations"] = evaluation
            with self.cache_lock:
                gateway = {} if sealed else dict(self.gateway_cache.get(name, {}))
            result["gateway"] = {**gateway, "enabled": True} if self.gateway_status and not sealed else {"enabled": False}
            if gateway.get("available") or (gateway.get("observed_at") and "evaluations" in gateway):
                result["evaluations"] = gateway.get("evaluations", [])
                result["evaluation_source"] = "Gateway ledger snapshot at " + gateway["observed_at"] + " (poll interval 30 seconds; not a live worker count)"
                if not gateway.get("available"):
                    result["evaluation_source"] += "; gateway poll pending or unavailable; showing last successful snapshot"
                by_id = {item.get("evaluation_id"): item for item in result["evaluations"]
                         if isinstance(item, dict) and hexadecimal(item.get("evaluation_id"))}
                for split, anchor in anchors.items():
                    latest = by_id.get(anchor.get("evaluation_id"))
                    if (latest and not (anchor["status"] == "complete" and anchor["status_source"] == "saved anchor result")
                            and latest.get("split") == split
                            and latest.get("candidate_commit") == result["initial_candidate"]
                            and latest.get("status") != "unknown"):
                        # Ledger completion does not establish numerical validity
                        # or readiness: those still require the saved full result.
                        anchor.update(status=latest["status"], status_source="gateway ledger snapshot",
                                      observed_at=gateway["observed_at"])
                    if latest and latest.get("split") == split and latest.get("candidate_commit") == result["initial_candidate"]:
                        if "progress" in latest:
                            anchor["progress"] = latest["progress"]
            elif self.gateway_status and not sealed:
                result["evaluation_source"] += "; gateway poll pending or unavailable"
            for item in result["evaluations"]:
                if item.get("evaluation_id") and item.get("submitted_at"):
                    submissions[item["evaluation_id"]] = item
            for point in result["research_progress"]["points"]:
                for split in ("train", "valid"):
                    submission = submissions.get(point.get(split + "_evaluation_id"), {})
                    point[split + "_submitted_at"] = (submission.get("submitted_at")
                        if submission.get("candidate_commit") == point.get("candidate_commit") else None)
            recovery = runtime.get("recover_evaluation_ids", [])
            if not isinstance(recovery, list):
                recovery = []
            result["recovery"] = []
            for identity in recovery[:MAX_FILES]:
                if hexadecimal(identity):
                    recovered = files.json(f"workspace/.evaluation_state/recovered/{identity}.json")
                    result["recovery"].append({"evaluation_id": identity,
                        "status": status(recovered.get("status")) if recovered.get("run_id") == saved["run_id"] else "awaiting recovery"})
            ralph = files.json("workspace/.ralph/ralph-loop.state.json") if purpose == "research" else {}
            result["execution"] = ("One-shot provider verification; Ralph is not used" if purpose == "verification"
                else "Ralph running (saved loop state)" if ralph.get("active") is True and container.get("running")
                else "Ralph loop state saved; container is not confirmed running" if ralph else "Ralph has not recorded a loop state")
            result["ralph_iteration"] = ralph.get("iteration") if type(ralph.get("iteration")) is int else None
            result["verification_complete"] = (files.json("workspace/verification/complete.json").get("status") == "verification_complete")
            invocation = files.json("workspace/verification/invocation.json")
            result["provider_invocations"] = (1 if invocation.get("run_id") == saved["run_id"] else 0) if purpose == "verification" else None
            if sealed and purpose == "verification" and not invocation:
                result["provider_invocations"] = None
            if container.get("phase") == "sign_in" and container.get("running"):
                result["execution"] = ("Sign-in in progress; verification provider session has not started"
                    if purpose == "verification" else "Sign-in in progress; Ralph has not started")
            result["anchors_state"] = ("ready" if result["anchors_ready"] else "not_started"
                if not any(item["started"] for item in anchors.values())
                else "pending_or_invalid")
            invalid = [split for split, item in anchors.items() if item.get("is_valid") == 0]
            if invalid:
                result["anchors_state"] = "invalid"
                result["gate"] = "Blocked: " + "; ".join(
                    split + " anchor invalid (" + anchors[split]["invalid_reason"] + ")" for split in invalid)
            if sealed:
                result["execution"] = ("Finished run" if saved["status"] == "finished" else "Run finishing") + "; showing verified exported diagnostics"
                result["evaluation_source"] = "Historical results from the verified export"
                if saved.get("retired_dataset"):
                    result["execution"] = "Retired dataset; showing historical results"
                if not invalid:
                    result["gate"] = "Finished; research is stopped"
                if not any(item["started"] for item in anchors.values()):
                    result["anchors_state"] = "unavailable"
                    for item in anchors.values():
                        item.update(status="unavailable", status_source="not recorded in verified export")
            logs = self.logs(files, purpose)
            result["logs"] = (["docker"] if container.get("owned") else []) + list(logs)
            activity = []
            for path in [*logs.values(), "workspace/.run-ready.json", "workspace/anchors/train.json",
                         "workspace/anchors/valid.json", "workspace/.evaluation_state/client.json",
                         "workspace/.evaluation_state/anchors.json", "workspace/verification/invocation.json"]:
                found = files.read(path, tail=True, limit=1)
                if found:
                    activity.append(found[1])
            result["last_activity"] = utc(max(activity)) if activity else result["updated_at"]
            result["last_activity_source"] = "Newest saved log/receipt modification; Docker output may be newer"
            if sealed:
                result["last_activity_source"] = "Export modification time; this view is a historical snapshot"
            # Apply known-secret redaction after projection too: model/config
            # metadata must never become an alternate credential-serving route.
            return self.redacted_value(result, () if sealed else secrets_for(files))

    @staticmethod
    def evaluation(raw, run_id):
        if not isinstance(raw, dict) or raw.get("run_id") not in {None, run_id}:
            return {"status": "unknown"}
        result = {"evaluation_id": hexadecimal(raw.get("evaluation_id")), "status": status(raw.get("status")),
                "submitted_at": timestamp(raw.get("timestamp")),
                "split": raw.get("split") if raw.get("split") in {"train", "valid", "test"} else None,
                "candidate_commit": hexadecimal(raw.get("candidate_commit")),
                "infrastructure_error": bool(raw.get("infrastructure_error"))}
        if "progress" in raw:
            result["progress"] = Monitor.progress(raw["progress"])
        return result

    @staticmethod
    def progress(raw):
        """Project durable task counts, never incomplete energy/step scores."""
        choices = {"available", "missing", "invalid", "oversized", "unsafe", "unreadable"}
        if (not isinstance(raw, dict) or not isinstance(raw.get("availability"), str)
                or raw["availability"] not in choices):
            return {"availability": "invalid"}
        result = {"availability": raw["availability"]}
        reason = raw.get("reason")
        if isinstance(reason, str) and re.fullmatch(r"[a-z_]{1,80}", reason):
            result["reason"] = reason
        if result["availability"] == "available":
            counts = {key: raw.get(key) for key in ("total", "completed", "pending")}
            if (not all(type(value) is int and 0 <= value <= 10**9 for value in counts.values())
                    or counts["completed"] + counts["pending"] != counts["total"]):
                return {"availability": "invalid"}
            result.update(counts)
            for key in ("attempts", "infrastructure_retries"):
                value = raw.get(key)
                result[key] = value if type(value) is int and 0 <= value <= 10**9 else None
        result["updated_at"] = timestamp(raw.get("updated_at"))
        result["age_seconds"] = (raw["age_seconds"] if type(raw.get("age_seconds")) in {int, float}
                                 and math.isfinite(raw["age_seconds"]) and 0 <= raw["age_seconds"] <= 10**10 else None)
        result["stale"] = raw.get("stale") if type(raw.get("stale")) is bool else None
        threshold = raw.get("stale_after_seconds")
        result["stale_after_seconds"] = threshold if type(threshold) is int and 0 < threshold <= 86400 else None
        return result

    def log(self, name, key):
        with self.run(name) as (files, saved):
            sealed = saved.get("status") in SEALED_STATES
            if sealed:
                if key == "docker":
                    raise MonitorError("finished logs require the verified export")
                try:
                    files = self.exported_files(files, saved)
                except (OSError, ValueError, EOFError, tarfile.TarError, RecursionError) as exc:
                    raise MonitorError("sanitized export unavailable") from exc
            if key == "docker":
                if not self.container(saved).get("owned"):
                    raise MonitorError("owned Docker container is unavailable; select a persisted log")
                reply = self.docker("logs", "--timestamps", "--tail", "500", "ar-agent-" + saved["run_id"], check=False)
                if reply.returncode:
                    raise MonitorError("Docker logs unavailable; select a persisted log")
                text = (reply.stdout + reply.stderr)[-MAX_TAIL:]
                heading = "Live Docker tail (last 500 lines, at most 256 KiB).\n"
            else:
                path = self.logs(files, saved.get("purpose", "research")).get(key)
                if path is None:
                    raise MonitorError("unknown log")
                found = files.read(path, tail=True, limit=MAX_TAIL)
                if found is None:
                    raise MonitorError("log unavailable")
                text = found[0].decode("utf-8", "replace")
                if found[2]:
                    # A tail may begin inside a credential or terminal escape;
                    # do not expose a partially redacted first line.
                    text = text.partition("\n")[2]
                heading = ("Persisted log tail (256 KiB; earlier content omitted).\n" if found[2] else "Persisted log.\n")
            if sealed:
                heading = "Verified sanitized export. " + heading
            return heading + redact(text, () if sealed else secrets_for(files))

    def table(self, name, key):
        if key not in TABLE_NAMES:
            raise MonitorError("unknown table")
        with self.run(name) as (files, saved):
            sealed = saved.get("status") in SEALED_STATES
            if sealed:
                try:
                    files = self.exported_files(files, saved)
                except (OSError, ValueError, EOFError, tarfile.TarError, RecursionError) as exc:
                    raise MonitorError("sanitized export unavailable") from exc
            result = files.table(key)
            result["name"] = key + ".tsv"
            result["source"] = "Verified sanitized export" if sealed else "Current run workspace"
            return self.redacted_value(result, () if sealed else secrets_for(files))


from isolated_runs.monitor_page import PAGE


class Handler(BaseHTTPRequestHandler):
    server_version = "AutoresearchMonitor"

    def log_message(self, *args):
        pass  # Query strings and untrusted log data never enter server logs.

    def send(self, code, body, content_type, *, download=None):
        body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        authority = f"127.0.0.1:{self.server.server_port}"
        # Strict authority and browser-origin checks prevent rebinding and
        # cross-site reads; deliberately no CORS or wildcard host support.
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if (hosts != [authority] or len(origins) > 1 or (origins and origins != ["http://" + authority])
                or fetch_site not in {None, "none", "same-origin"}):
            self.send(403, "Local operator origin required.\n", "text/plain")
            return
        try:
            referrer = self.headers.get("Referer")
            if referrer and (urlsplit(referrer).scheme, urlsplit(referrer).netloc) != ("http", authority):
                self.send(403, "Local operator origin required.\n", "text/plain")
                return
            parsed = urlsplit(self.path)
            if len(self.path) > 2048 or parsed.scheme or parsed.netloc or parsed.fragment:
                raise MonitorError("unknown route")
            query = parse_qs(parsed.query, strict_parsing=True, max_num_fields=3)
            if any(len(v) != 1 for v in query.values()):
                raise MonitorError("invalid query")
            if parsed.path == "/" and not query:
                self.send(200, PAGE, "text/html")
            elif parsed.path == "/api/runs" and not query:
                self.send(200, json.dumps({"runs": self.server.monitor.names(),
                    "display_names": self.server.monitor.display_names(),
                    "gateway_status_enabled": self.server.monitor.gateway_status,
                    "demo": getattr(self.server.monitor, "demo", False)}), "application/json")
            elif parsed.path == "/api/run" and set(query) == {"name"}:
                self.send(200, json.dumps(self.server.monitor.detail(query["name"][0])), "application/json")
            elif parsed.path == "/log" and set(query) == {"name", "log"}:
                self.send(200, self.server.monitor.log(query["name"][0], query["log"][0]), "text/plain")
            elif parsed.path == "/api/table" and set(query) == {"name", "table"}:
                self.send(200, json.dumps(self.server.monitor.table(query["name"][0], query["table"][0])), "application/json")
            elif parsed.path == "/table" and set(query) in ({"name", "table"}, {"name", "table", "download"}):
                if "download" in query and query["download"] != ["1"]:
                    raise MonitorError("invalid download option")
                table = self.server.monitor.table(query["name"][0], query["table"][0])
                if table["raw"] is None:
                    raise MonitorError("raw bounded table unavailable")
                self.send(200, table["raw"], "text/plain", download=table["name"] if "download" in query else None)
            else:
                raise MonitorError("unknown route")
        except (MonitorError, OSError, ValueError, TypeError):
            self.send(404, "Requested run or bounded log is unavailable.\n", "text/plain")

    def do_POST(self):
        self.send(405, "Read-only operator view.\n", "text/plain")


class Server(HTTPServer):
    allow_reuse_address = True

    def __init__(self, monitor, port=8765):
        self.monitor = monitor
        super().__init__(("127.0.0.1", port), Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="label this separate generated-data viewer as a demonstration")
    parser.add_argument("--gateway-status", action="store_true",
                        help="poll the private configured admin status argv in the background every 30 seconds")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    monitor = None
    try:
        monitor = Monitor(args.root, gateway_status=args.gateway_status)
        monitor.demo = args.demo
        with Server(monitor, args.port) as server:
            print(f"Autoresearch operator view: http://127.0.0.1:{server.server_port}/", flush=True)
            server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    except (OSError, MonitorError):
        print("Monitor could not open the private run root or loopback port.", flush=True)
        return 2
    finally:
        if monitor:
            monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
