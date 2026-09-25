from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import shutil

import pytest

from isolated_runs.relay import RelayError, SSHRelay, validate_relay


@pytest.fixture
def relay(tmp_path):
    class SSH:
        def __init__(self):
            self.calls = []
            self.sockets = {}
            self.fail_exit = False
            self.fail_forward = False

        def __call__(self, argv, *, timeout, check=True, capture=True):
            self.calls.append((argv, timeout))
            path = argv[argv.index("-S") + 1]
            result = 0
            if "-N" in argv:
                assert capture is False
                sock = socket.socket(socket.AF_UNIX)
                sock.bind(path)
                sock.listen()
                self.sockets[path] = sock
            else:
                action = argv[argv.index("-O") + 1]
                if action == "exit":
                    if self.fail_exit:
                        result = 1
                    else:
                        self.sockets.pop(path).close()
                        Path(path).unlink()
                if action == "forward" and self.fail_forward:
                    result = 1
            return subprocess.CompletedProcess(argv, result, "", "")

    command = SSH()
    instance = SSHRelay(tmp_path, "own-run-a123", {"port": 42391,
        "relay": {"ssh_target": "cpu-33-via-golf", "target_host": "127.0.0.1", "target_port": 22}}, command)
    yield instance, command
    command.fail_exit = False
    # Tests that deliberately alter ownership restore the marker themselves.
    instance.stop()


def test_host_master_has_one_fixed_loopback_forward_and_no_inherited_sessions(relay):
    instance, command = relay
    instance.start()
    assert instance.status() == {"state": "running"}
    master = command.calls[0][0]
    assert "-N" in master and "-T" in master and "ClearAllForwardings=yes" in master
    assert master[master.index("-E") + 1] == str(instance.log_path)
    assert all(option in master for option in ("ForwardAgent=no", "ForwardX11=no", "GatewayPorts=no",
                                               "PermitLocalCommand=no", "RemoteCommand=none"))
    assert "-L" not in master and "-R" not in master and "-D" not in master
    forward = next(argv for argv, _ in command.calls if "forward" in argv)
    assert forward[1:3] == ["-F", "/dev/null"]
    assert forward[forward.index("-L") + 1] == "127.0.0.1:42391:127.0.0.1:22"
    assert all(0 < timeout <= 25 for _, timeout in command.calls)
    directory = Path(json.loads(instance.metadata.read_text())["directory"])
    assert directory.stat().st_mode & 0o077 == 0
    assert instance.metadata.stat().st_mode & 0o077 == 0
    instance.stop()
    assert not directory.exists() and not instance.metadata.exists()
    assert instance.status() == {"state": "stopped"}


def test_ssh_log_retains_bounded_tail_and_stays_writable_by_open_append_fd(relay):
    instance, _ = relay
    instance.cap_log()
    with instance.log_path.open("ab", buffering=0) as writer:
        writer.write(b"discarded history\n" + b"x" * 65536)
        instance.cap_log()
        assert instance.log_path.read_bytes() == b"x" * 65536
        writer.write(b"\nnew disconnect\n")
        instance.cap_log()
        assert instance.log_path.stat().st_size == 65536
        assert instance.log_path.read_bytes().endswith(b"\nnew disconnect\n")
    assert instance.log_path.stat().st_mode & 0o777 == 0o600
    instance.start()
    instance.stop()
    assert instance.log_path.read_bytes().endswith(b"\nnew disconnect\n")


def test_ssh_log_does_not_follow_symlink(relay, tmp_path):
    instance, _ = relay
    outside = tmp_path / "unrelated.log"
    outside.write_text("preserve")
    instance.log_path.symlink_to(outside)
    with pytest.raises(OSError):
        instance.cap_log()
    assert outside.read_text() == "preserve"


@pytest.mark.skipif(not shutil.which("ssh"), reason="OpenSSH client required")
def test_failed_real_ssh_start_preserves_transport_error_without_network(tmp_path):
    from isolated_runs.launcher import run_cmd

    def disconnected_proxy(argv, **kwargs):
        if "-N" in argv:
            # Real SSH, but the proxy exits locally before any network access.
            argv = ["ssh", "-F", "/dev/null", "-o", "ProxyCommand=" + shutil.which("false"), *argv[1:]]
        return run_cmd(argv, **kwargs)

    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    instance = SSHRelay(tmp_path, "local-failure-test", {"port": port,
        "relay": {"ssh_target": "unused.invalid", "target_host": "127.0.0.1", "target_port": 22}},
        disconnected_proxy)
    with pytest.raises(RelayError):
        instance.start()
    assert not instance.metadata.exists()
    diagnostic = instance.log_path.read_text().lower()
    assert "closed" in diagnostic or "reset" in diagnostic or "broken pipe" in diagnostic


def test_resume_replaces_only_its_owned_master(relay):
    instance, command = relay
    instance.start()
    old = instance._directory()
    instance.start()
    assert old != instance._directory() and not old.exists()
    assert len([argv for argv, _ in command.calls if "-N" in argv]) == 2
    assert len([argv for argv, _ in command.calls if "exit" in argv]) == 1


def test_failed_forward_removes_master_and_failed_exit_preserves_ownership(relay):
    instance, command = relay
    command.fail_forward = True
    with pytest.raises(RelayError, match="forward was not acknowledged"):
        instance.start()
    assert not instance.metadata.exists() and not command.sockets
    command.fail_forward = False
    instance.start()
    command.fail_exit = True
    with pytest.raises(RelayError, match="exit was not acknowledged"):
        instance.stop()
    assert instance.metadata.exists() and command.sockets
    starts = len([argv for argv, _ in command.calls if "-N" in argv])
    with pytest.raises(RelayError):
        instance.start()
    assert len([argv for argv, _ in command.calls if "-N" in argv]) == starts


def test_relay_never_controls_another_runs_socket(relay):
    instance, command = relay
    instance.start()
    marker = instance._directory() / "owner.json"
    original = marker.read_text()
    marker.write_text(json.dumps({"run_id": "someone-else"}))
    count = len(command.calls)
    try:
        with pytest.raises(RelayError, match="another run"):
            instance.stop()
        assert len(command.calls) == count
    finally:
        marker.write_text(original)


def test_dead_master_stale_socket_recovers_without_pid_signals(relay):
    instance, command = relay
    instance.start()
    old = instance._directory()
    command.sockets.pop(str(old / "control")).close()
    command.fail_exit = True
    instance.stop()
    assert not old.exists()
    command.fail_exit = False
    instance.start()
    assert instance.status()["state"] == "running"


def test_missing_socket_with_occupied_port_never_removes_ownership(relay):
    instance, command = relay
    instance.start()
    directory = instance._directory()
    command.sockets.pop(str(directory / "control")).close()
    (directory / "control").unlink()
    with socket.socket() as unrelated:
        unrelated.bind(("127.0.0.1", 0))
        unrelated.listen()
        instance.evaluation["port"] = unrelated.getsockname()[1]
        with pytest.raises(RelayError, match="occupied"):
            instance.stop()
        assert instance.metadata.exists() and directory.exists()
    instance.stop()
    assert not instance.metadata.exists()


@pytest.mark.parametrize("change", [
    {"ssh_target": "-oProxyCommand=evil"}, {"target_host": "localhost:22:elsewhere"},
    {"target_host": "$(touch /tmp/x)"}, {"target_port": True}, {"target_port": 0},
    {"local_bind": "0.0.0.0"}, {"ssh_command": ["arbitrary"]},
])
def test_relay_config_refuses_extra_routes_and_shell_fields(change):
    with pytest.raises(RelayError):
        validate_relay({"ssh_target": "cpu-33-via-golf", "target_host": "127.0.0.1", "target_port": 22, **change})
