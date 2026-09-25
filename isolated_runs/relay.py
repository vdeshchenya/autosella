"""Host-only, run-owned fixed SSH relay for a local Docker Desktop engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
import time


class RelayError(RuntimeError):
    pass


def validate_relay(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != {"ssh_target", "target_host", "target_port"}:
        raise RelayError("evaluation.relay requires ssh_target, target_host, target_port")
    for key in ("ssh_target", "target_host"):
        if not isinstance(config[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config[key]):
            raise RelayError(f"relay.{key} must be an explicit host name, IPv4 address or SSH alias")
    if type(config["target_port"]) is not int or not 1 <= config["target_port"] <= 65535:
        raise RelayError("invalid relay target port")


class SSHRelay:
    """Control only a private multiplex socket; never kill a remembered PID.

    The host's SSH config supplies authentication and jump routing. A fresh
    master clears *all* inherited forwards. A second config-free mux command
    installs only the explicit evaluator TCP forward. No general SSH session
    or host credential is made available to the research container.
    """

    def __init__(self, control: Path, run_id: str, evaluation: dict, command):
        self.metadata = control / "relay.json"
        self.log_path = control / "relay-ssh.log"
        self.run_id, self.evaluation, self.command = run_id, evaluation, command
        self.config = evaluation["relay"]
        validate_relay(self.config)

    def cap_log(self, limit: int = 65536) -> None:
        """Keep a private tail, including while ssh holds its append fd open."""
        fd = os.open(self.log_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        with os.fdopen(fd, "r+b") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise RelayError("SSH log must be an owned regular file")
            os.fchmod(handle.fileno(), 0o600)
            if info.st_size > limit:
                handle.seek(-limit, os.SEEK_END)
                tail = handle.read(limit)
                handle.seek(0)
                handle.write(tail)
                handle.truncate()

    def _directory(self) -> Path | None:
        if self.metadata.is_symlink():
            raise RelayError("relay metadata must not be a symlink")
        if not self.metadata.exists():
            return None
        record = json.loads(self.metadata.read_text())
        directory = Path(record["directory"])
        if record.get("run_id") != self.run_id or directory.parent != Path("/tmp"):
            raise RelayError("relay ownership record does not match this run")
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RelayError("relay directory ownership or permissions changed")
        marker = directory / "owner.json"
        if marker.is_symlink() or json.loads(marker.read_text()) != {"run_id": self.run_id}:
            raise RelayError("refusing an SSH relay owned by another run")
        sock = directory / "control"
        if sock.is_symlink() or (sock.exists() and not stat.S_ISSOCK(sock.lstat().st_mode)):
            raise RelayError("relay control path is not an owned Unix socket")
        return directory

    def _mux(self, directory: Path, action: str, *args: str):
        return self.command(["ssh", "-F", "/dev/null", "-S", str(directory / "control"),
                             "-o", "BatchMode=yes", "-O", action, *args,
                             self.config["ssh_target"]], check=False, timeout=10)

    def status(self) -> dict:
        directory = self._directory()
        if directory is None:
            return {"state": "stopped"}
        if not (directory / "control").exists():
            return {"state": "unavailable", "detail": "owned SSH control socket is absent"}
        result = self._mux(directory, "check")
        return {"state": "running" if result.returncode == 0 else "unavailable"}

    def start(self) -> None:
        # Reconcile leftovers before replacing a tunnel. No shared masters,
        # guessed PID kills, stale listeners or silent forwarding reuse.
        self.stop()
        self.cap_log()
        directory = Path(tempfile.mkdtemp(prefix="ar-relay-", dir="/tmp"))
        (directory / "owner.json").write_text(json.dumps({"run_id": self.run_id}))
        self.metadata.write_text(json.dumps({"run_id": self.run_id, "directory": str(directory)}))
        self.metadata.chmod(0o600)
        options = ["BatchMode=yes", "ControlMaster=yes", "ControlPersist=no", "ClearAllForwardings=yes",
                   "ForwardAgent=no", "ForwardX11=no", "GatewayPorts=no", "PermitLocalCommand=no", "RemoteCommand=none",
                   "StrictHostKeyChecking=yes", "ConnectionAttempts=1", "ConnectTimeout=10",
                   "ServerAliveInterval=15", "ServerAliveCountMax=2", "ExitOnForwardFailure=yes"]
        argv = ["ssh", "-N", "-T", "-f", "-E", str(self.log_path), "-S", str(directory / "control")]
        for value in options:
            argv += ["-o", value]
        argv += [self.config["ssh_target"]]
        try:
            # ssh -f may leave stderr open in its detached master/proxy. Pipes
            # would keep communicate() waiting after successful daemonization.
            self.command(argv, timeout=25, capture=False)
            forward = f'127.0.0.1:{self.evaluation["port"]}:{self.config["target_host"]}:{self.config["target_port"]}'
            if self._mux(directory, "forward", "-L", forward).returncode:
                raise RelayError("fixed evaluator relay forward was not acknowledged")
            if self.status()["state"] != "running":
                raise RelayError("evaluator relay master did not become ready")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        directory = self._directory()
        if directory is None:
            return
        sock = directory / "control"
        if sock.exists():
            if self._mux(directory, "exit").returncode:
                # A killed master can leave a stale socket. Prove both the
                # Unix endpoint and loopback listener are dead before removing
                # this run's own stale path; never infer absence from mux rc.
                try:
                    with socket.socket(socket.AF_UNIX) as connection:
                        connection.settimeout(.3)
                        connection.connect(str(sock))
                except ConnectionRefusedError:
                    self._require_closed_port()
                    sock.unlink()
                else:
                    raise RelayError("owned SSH relay exit was not acknowledged")
            deadline = time.monotonic() + 3
            while sock.exists() and time.monotonic() < deadline:
                time.sleep(.05)
            if sock.exists():
                raise RelayError("owned SSH relay socket remained after exit")
        else:
            # A dead master normally removes its socket and listener. If a
            # listener remains, ownership cannot be proved: preserve state and
            # fail closed instead of touching a possibly unrelated process.
            self._require_closed_port()
        (directory / "owner.json").unlink()
        directory.rmdir()
        self.metadata.unlink()

    def _require_closed_port(self) -> None:
        try:
            with socket.create_connection(("127.0.0.1", self.evaluation["port"]), timeout=.3):
                raise RelayError("relay socket is absent but its local port is occupied; cleanup is uncertain")
        except ConnectionRefusedError:
            pass
