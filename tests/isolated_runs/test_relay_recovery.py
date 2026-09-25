from __future__ import annotations

import fcntl
import json
import os
from types import SimpleNamespace

import pytest

from isolated_runs import relay_recovery as recovery
from isolated_runs.launcher import Launcher, LaunchError, LifecycleBusy
from isolated_runs.relay import RelayError


NAME = "trial-a"
RUN_ID = "trial-a-owned-run"


class Events(list):
    def info(self, message):
        self.append(json.loads(message))


class FakeRelay:
    def __init__(self):
        self.running = True
        self.fail_start = False
        self.starts = 0
        self.status_calls = 0
        self.cap_calls = 0
        self.during_status = lambda: None
        self.during_start = lambda: None

    def cap_log(self):
        self.cap_calls += 1

    def status(self):
        self.status_calls += 1
        self.during_status()
        return {"state": "running" if self.running else "unavailable"}

    def start(self):
        self.starts += 1
        self.during_start()
        if self.fail_start:
            raise RelayError("simulated SSH transport failure")
        self.running = True


@pytest.fixture
def setup(tmp_path):
    class TestLauncher(Launcher):
        def __init__(self, root):
            super().__init__(root)
            self.transport = FakeRelay()
            self.state_reads = 0

        def state(self, name):
            self.state_reads += 1
            return super().state(name)

        def relay(self, state):
            return self.transport

        def docker(self, *args, **kwargs):
            raise AssertionError("relay recovery must not change research containers")

        def admin(self, *args, **kwargs):
            raise AssertionError("relay recovery must not change or submit evaluations")

    launcher = TestLauncher(tmp_path / "runs")
    control = launcher.path(NAME) / "control"
    control.mkdir(parents=True)
    (control / "config.json").write_text(json.dumps({"evaluation": {"relay": {
        "ssh_target": "unused-test-host", "target_host": "127.0.0.1", "target_port": 22}}}))
    launcher.save(NAME, {"name": NAME, "run_id": RUN_ID, "status": "running", "key_revoked": False})
    return launcher, launcher.transport, control, Events()


def tick(setup, retry, now, **kwargs):
    launcher, _, _, events = setup
    return recovery.recover_once(launcher, NAME, RUN_ID, retry, events, clock=lambda: now, **kwargs)


def run_worker(launcher):
    # Production is a separate process; do not leak its umask into other tests.
    previous = os.umask(0o077)
    try:
        return recovery.run(launcher, NAME, RUN_ID)
    finally:
        os.umask(previous)


def test_healthy_relay_is_adopted_without_changing_research_or_run_state(setup):
    launcher, relay, control, events = setup
    original = (control / "state.json").read_bytes()
    retry = recovery.Retry()
    assert tick(setup, retry, 0)
    assert tick(setup, retry, 60)
    assert relay.starts == 0 and relay.status_calls == 2
    assert relay.cap_calls == 2
    assert (control / "state.json").read_bytes() == original
    assert not events


def test_repeated_failures_back_off_without_repeated_connection_attempts(setup):
    _, relay, _, events = setup
    relay.running = False
    relay.fail_start = True
    retry = recovery.Retry()
    for now, expected in [(0, 1), (4.9, 1), (5, 2), (14.9, 2), (15, 3),
                          (34.9, 3), (35, 4), (74.9, 4), (75, 5), (134.9, 5), (135, 6)]:
        assert tick(setup, retry, now)
        assert relay.starts == expected
    failed = [e for e in events if e["event"] == "relay_recovery_failed"]
    assert [e["retry_in_seconds"] for e in failed] == [5, 10, 20, 40, 60, 60]
    assert relay.status_calls == 6


def test_brief_recoveries_preserve_backoff_until_a_stable_minute(setup):
    _, relay, _, events = setup
    relay.running = False
    retry = recovery.Retry()
    assert tick(setup, retry, 0)
    relay.running = False
    assert tick(setup, retry, 4)
    assert relay.starts == 1
    assert tick(setup, retry, 5)
    relay.running = False
    assert tick(setup, retry, 6)
    assert tick(setup, retry, 14)
    assert relay.starts == 2
    assert tick(setup, retry, 15)
    assert tick(setup, retry, 74)
    assert retry.delay == 40
    assert tick(setup, retry, 75)
    assert retry.delay == 5
    relay.running = False
    assert tick(setup, retry, 76)
    restarted = [e for e in events if e["event"] == "relay_restarted"]
    assert [e["retry_delay_seconds"] for e in restarted] == [5, 10, 20, 5]


@pytest.mark.parametrize("error", [RelayError("ownership is uncertain"),
                                  OSError("control socket could not be inspected"),
                                  ValueError("invalid ownership record")])
def test_status_errors_back_off_without_blindly_replacing_the_relay(setup, error):
    _, relay, _, events = setup

    def unavailable():
        raise error

    relay.during_status = unavailable
    retry = recovery.Retry()
    assert tick(setup, retry, 0)
    assert tick(setup, retry, 4)
    assert relay.status_calls == 1 and relay.starts == 0
    assert events[-1]["event"] == "relay_recovery_failed"
    assert events[-1]["error_type"] == type(error).__name__
    relay.during_status = lambda: None
    assert tick(setup, retry, 5)
    assert relay.starts == 0 and relay.status_calls == 2


@pytest.mark.parametrize("change", [
    {"status": status} for status in
    ("created", "starting", "stopping", "stopped", "stopped_unconfirmed", "failed", "finishing", "finished")
] + [{"key_revoked": True}, {"run_id": "another-owned-run"}])
def test_inactive_revoked_or_replaced_runs_are_never_restarted(setup, change):
    launcher, relay, _, events = setup
    launcher.save(NAME, {**launcher.state(NAME), **change})
    relay.running = False
    inactive = []
    assert tick(setup, recovery.Retry(), 0, on_inactive=lambda: inactive.append(True)) is False
    assert inactive == [True]
    assert relay.starts == relay.status_calls == relay.cap_calls == 0
    assert events[-1]["event"] == "supervisor_stopped"


def test_state_read_status_and_repair_share_the_real_lifecycle_lock(setup):
    launcher, relay, _, _ = setup
    contender = Launcher(launcher.root)

    def assert_locked():
        with pytest.raises(LifecycleBusy), contender.lock(NAME):
            pytest.fail("repair released the lifecycle lock")

    relay.during_status = relay.during_start = assert_locked
    relay.running = False
    assert tick(setup, recovery.Retry(), 0)
    assert relay.starts == 1

    # The supervisor cannot read running while stop owns the lock, then later
    # restart from that stale observation after the operator writes stopping.
    relay.running = False
    with contender.lock(NAME):
        reads = launcher.state_reads
        with pytest.raises(LifecycleBusy):
            tick(setup, recovery.Retry(), 1)
        assert launcher.state_reads == reads
        launcher.save(NAME, {**launcher.state(NAME), "status": "stopping"})
    assert tick(setup, recovery.Retry(), 2) is False
    assert relay.starts == 1


def test_singleton_lease_prevents_a_second_worker_and_keeps_the_same_inode(setup, monkeypatch):
    launcher, _, control, _ = setup
    with recovery.open_lease(control) as lease:
        inode = os.fstat(lease.fileno()).st_ino
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert recovery.supervisor_running(control)
        monkeypatch.setattr(recovery, "recover_once", lambda *a, **k: pytest.fail("duplicate worker reconciled"))
        assert run_worker(launcher) == 0
        assert recovery.supervisor_running(control)
    assert not recovery.supervisor_running(control)
    assert (control / "relay-recovery.lock").stat().st_ino == inode


def test_inactive_worker_releases_its_lease_before_unlocking_lifecycle(setup, monkeypatch):
    launcher, _, control, _ = setup
    launcher.save(NAME, {**launcher.state(NAME), "status": "stopped"})
    original = recovery.recover_once
    observed = []

    def inspect_release(*args, **kwargs):
        close_lease = kwargs["on_inactive"]

        def release():
            assert recovery.supervisor_running(control)
            close_lease()
            assert not recovery.supervisor_running(control)
            with pytest.raises(LifecycleBusy), launcher.lock(NAME):
                pytest.fail("resume could race an exiting supervisor")
            observed.append(True)

        return original(*args, **{**kwargs, "on_inactive": release})

    monkeypatch.setattr(recovery, "recover_once", inspect_release)
    assert run_worker(launcher) == 0
    assert observed == [True]


def test_worker_defers_busy_lifecycle_then_reads_the_completed_stop(setup, monkeypatch):
    launcher, relay, control, _ = setup
    relay.running = False
    operator = launcher.lock(NAME)
    operator.__enter__()
    sleeps = []

    def finish_stop(delay):
        sleeps.append(delay)
        assert relay.starts == 0
        launcher.save(NAME, {**launcher.state(NAME), "status": "stopped"})
        operator.__exit__(None, None, None)

    monkeypatch.setattr(recovery.time, "sleep", finish_stop)
    try:
        assert run_worker(launcher) == 0
    finally:
        if not sleeps:
            operator.__exit__(None, None, None)
    assert sleeps == [recovery.POLL_SECONDS]
    assert relay.starts == 0
    events = [json.loads(line) for line in (control / "relay-recovery.log").read_text().splitlines()]
    assert [e["event"] for e in events] == ["supervisor_started", "supervisor_stopped"]


def test_startup_acknowledges_singleton_before_waiting_for_lifecycle(setup, monkeypatch):
    launcher, relay, control, _ = setup
    children = []
    leases = []

    def spawn(*args, **kwargs):
        with pytest.raises(LifecycleBusy), launcher.lock(NAME):
            pytest.fail("test must model the launching parent's lifecycle lock")
        lease = recovery.open_lease(control)
        leases.append(lease)
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = SimpleNamespace(poll=lambda: pytest.fail("acknowledged worker was polled"))
        children.append(child)
        return child

    monkeypatch.setattr(recovery.subprocess, "Popen", spawn)
    try:
        with launcher.lock(NAME):
            recovery.ensure_supervisor(launcher, NAME)
            recovery.ensure_supervisor(launcher, NAME)
        assert len(children) == 1
        assert recovery.supervisor_running(control)
        assert relay.starts == 0
    finally:
        for lease in leases:
            lease.close()


@pytest.mark.parametrize("early_exit", [True, False])
def test_failed_or_unacknowledged_startup_is_not_reported_as_protected(setup, monkeypatch, early_exit):
    launcher, relay, control, _ = setup
    code = 23 if early_exit else None
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **k:
                        SimpleNamespace(returncode=code, poll=lambda: code))
    times = iter([0, 1, 11])
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(recovery.time, "sleep", lambda delay: None)
    with launcher.lock(NAME), pytest.raises(LaunchError, match="did not start|not acknowledged"):
        recovery.ensure_supervisor(launcher, NAME)
    assert relay.starts == 0
    assert not recovery.supervisor_running(control)
