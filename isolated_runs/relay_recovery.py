"""Run-owned SSH recovery with backoff; never submits evaluation work."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from isolated_runs.launcher import Launcher, LaunchError, LifecycleBusy, ROOT
from isolated_runs.relay import RelayError

POLL_SECONDS = 5
MAX_BACKOFF = 60
STABLE_SECONDS = 60


def open_lease(control: Path):
    # Never unlink this inode: competing supervisors must lock the same file.
    fd = os.open(control / "relay-recovery.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(fd, "a")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        handle.close()
        raise LaunchError("relay recovery lock must be an owned regular file")
    os.fchmod(fd, 0o600)
    return handle


def supervisor_running(control: Path) -> bool:
    with open_lease(control) as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def ensure_supervisor(launcher: Launcher, name: str) -> None:
    """Called under the lifecycle lock, also usable for an already running run."""
    state = launcher.state(name)
    control = launcher.path(name) / "control"
    cfg = json.loads((control / "config.json").read_text())
    if state["status"] != "running" or state.get("key_revoked") or "relay" not in cfg["evaluation"]:
        raise LaunchError("relay recovery requires a running run with an enabled relay")
    if supervisor_running(control):
        return
    process = subprocess.Popen(
        [sys.executable, "-m", "isolated_runs.relay_recovery", "--root", str(launcher.root),
         "--name", name, "--run-id", state["run_id"]], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        # The child takes its lifetime lock BEFORE the lifecycle lock. Waiting
        # for its first repair here would deadlock against the launching parent.
        if supervisor_running(control):
            return
        if process.poll() is not None:
            raise LaunchError(f"relay recovery did not start (exit {process.returncode})")
        time.sleep(.05)
    raise LaunchError("relay recovery startup was not acknowledged")


def event(logger, kind: str, **fields) -> None:
    logger.info(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}))


@dataclass
class Retry:
    delay: float = POLL_SECONDS
    next_attempt: float = 0
    healthy_since: float | None = None

    def attempted(self, now: float) -> float:
        delay = self.delay
        self.next_attempt = now + delay
        self.delay = min(self.delay * 2, MAX_BACKOFF)
        return delay

    def healthy(self, now: float) -> None:
        if self.healthy_since is None:
            self.healthy_since = now
        if now - self.healthy_since >= STABLE_SECONDS:
            self.delay, self.next_attempt = POLL_SECONDS, 0


def recover_once(launcher, name, run_id, retry, logger, *, clock=time.monotonic, on_inactive=lambda: None):
    """Fresh intent and repair share one lock; no sleeps or jobs inside it."""
    with launcher.lock(name):
        state = launcher.state(name)
        if state["run_id"] != run_id or state["status"] != "running" or state.get("key_revoked"):
            event(logger, "supervisor_stopped", run_status=state["status"])
            # Release our lifetime lease before releasing the lifecycle lock.
            # Immediate resume cannot mistake an exiting supervisor for a live one.
            on_inactive()
            return False
        now = clock()
        if now < retry.next_attempt and retry.healthy_since is None:
            return True
        try:
            relay = launcher.relay(state)
            relay.cap_log()
            if relay.status()["state"] == "running":
                retry.healthy(now)
                return True
            retry.healthy_since = None
            if now < retry.next_attempt:
                return True
            event(logger, "relay_restart_attempt")
            relay.start()
            delay = retry.attempted(clock())
            retry.healthy_since = clock()
            event(logger, "relay_restarted", retry_delay_seconds=delay)
        except (RelayError, OSError, ValueError, KeyError) as exc:
            retry.healthy_since = None
            delay = retry.attempted(clock())
            event(logger, "relay_recovery_failed", error_type=type(exc).__name__,
                  detail=str(exc)[:384], retry_in_seconds=delay)
        return True


def run(launcher: Launcher, name: str, run_id: str) -> int:
    control = launcher.path(name) / "control"
    os.umask(0o077)
    logger = logging.Logger("relay-recovery")
    handler = RotatingFileHandler(control / "relay-recovery.log", maxBytes=65536, backupCount=2)
    logger.addHandler(handler)
    try:
        with open_lease(control) as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            event(logger, "supervisor_started", run_id=run_id, pid=os.getpid())
            retry = Retry()
            while True:
                try:
                    if not recover_once(launcher, name, run_id, retry, logger, on_inactive=lease.close):
                        return 0
                except LifecycleBusy:
                    pass  # Operator action owns the run; retry after it finishes.
                except (RelayError, OSError, ValueError, KeyError) as exc:
                    # Keep evidence of control-file problems without spinning.
                    event(logger, "supervisor_error", error_type=type(exc).__name__, detail=str(exc)[:384])
                    time.sleep(MAX_BACKOFF)
                time.sleep(POLL_SECONDS)
    finally:
        handler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    return run(Launcher(args.root), args.name, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
