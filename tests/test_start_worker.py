"""Tests for distributed_validate.worker.start_worker — the daemon loop.

Loops are broken out of by making ``load_next_optimization_task`` raise a
sentinel exception on the iteration where we want the test to stop. The
sentinel propagates up through ``while True`` uncaught (the try/except inside
the loop body only wraps the call to ``_run_optimization_task``).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_fakeredis


@pytest.fixture
def fake_redis_conn(monkeypatch):
    """Patch ``redis.Redis`` so start_worker uses a FakeStrictRedis instance."""
    import fakeredis
    import redis

    conn = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(redis, "Redis", lambda **kw: conn)
    return conn


class _LoopExit(Exception):
    """Sentinel used to break start_worker's ``while True``."""


def _canned_task():
    # optimizer_spec must normalize cleanly (start_worker logs
    # describe_optimizer_spec of it). Use a minimal importable shape —
    # nothing actually imports it because _run_optimization_task is stubbed.
    return "xtb:m1:abc123", {
        "mode": "xtb",
        "mol_name": "m1",
        "baseline": {},
        "max_steps": 50,
        "optimizer_spec": {"module_name": "sample_optimizer"},
    }


# ── happy path ──────────────────────────────────────────────────────────────

def test_start_worker_processes_task_and_stores_success(
    monkeypatch, fake_redis_conn, tmp_path
):
    """fetch a task → run it → store the result in Redis."""
    from distributed_validate import worker
    from distributed_validate.protocol import get_optimization_result

    monkeypatch.setattr(worker, "_preflight_backend_import", lambda mode: None)

    canned = {
        "result": {
            "mol_name": "m1",
            "converged": True,
            "max_steps": 50,
            "n_steps": 5,
            "rel_energy": 1.0,
            "rel_steps": 0.1,
        },
        "error": None,
    }
    monkeypatch.setattr(worker, "_run_optimization_task", lambda task: canned)

    task_id, task = _canned_task()
    calls = {"n": 0}

    def fake_load(redis_conn, modes, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return task_id, task
        raise _LoopExit("done")

    monkeypatch.setattr(worker, "load_next_optimization_task", fake_load)

    with pytest.raises(_LoopExit):
        worker.start_worker(
            redis_host="localhost",
            redis_port=6379,
            supported_modes=["xtb"],
            poll_interval_seconds=0.001,
            log_dir=str(tmp_path / "logs"),
            worker_name="happy_worker",
        )

    stored = get_optimization_result(fake_redis_conn, task_id)
    assert stored is not None
    assert stored["status"] == "complete"
    assert stored["data"] == canned


# ── loop catches task exceptions and records them ──────────────────────────

def test_start_worker_catches_task_exception_and_stores_error(
    monkeypatch, fake_redis_conn, tmp_path
):
    """If _run_optimization_task raises, the loop records a failure result
    instead of crashing."""
    from distributed_validate import worker
    from distributed_validate.protocol import get_optimization_result

    monkeypatch.setattr(worker, "_preflight_backend_import", lambda mode: None)

    def boom(task):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(worker, "_run_optimization_task", boom)

    task_id, task = _canned_task()
    calls = {"n": 0}

    def fake_load(redis_conn, modes, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return task_id, task
        raise _LoopExit("done")

    monkeypatch.setattr(worker, "load_next_optimization_task", fake_load)

    with pytest.raises(_LoopExit):
        worker.start_worker(
            redis_host="localhost",
            redis_port=6379,
            supported_modes=["xtb"],
            poll_interval_seconds=0.001,
            log_dir=str(tmp_path / "logs"),
            worker_name="fail_worker",
        )

    stored = get_optimization_result(fake_redis_conn, task_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["data"]["error"] == "backend exploded"
    assert stored["data"]["result"] is None


# ── polling when queue is empty ─────────────────────────────────────────────

def test_start_worker_polls_when_no_tasks(monkeypatch, fake_redis_conn, tmp_path):
    """When load returns (None, None), the worker sleeps and retries rather
    than exiting or running _run_optimization_task."""
    from distributed_validate import worker

    monkeypatch.setattr(worker, "_preflight_backend_import", lambda mode: None)

    run_calls = {"n": 0}

    def tracking_run(task):
        run_calls["n"] += 1
        return {"result": None, "error": None}

    monkeypatch.setattr(worker, "_run_optimization_task", tracking_run)

    poll_calls = {"n": 0}

    def fake_load(redis_conn, modes, **kwargs):
        poll_calls["n"] += 1
        if poll_calls["n"] < 3:
            return None, None
        raise _LoopExit("exit after 3 polls")

    monkeypatch.setattr(worker, "load_next_optimization_task", fake_load)

    with pytest.raises(_LoopExit):
        worker.start_worker(
            redis_host="localhost",
            redis_port=6379,
            supported_modes=["xtb"],
            poll_interval_seconds=0.001,
            log_dir=str(tmp_path / "logs"),
            worker_name="poll_worker",
        )

    assert poll_calls["n"] == 3
    # _run_optimization_task is never called for None tasks
    assert run_calls["n"] == 0


# ── preflight errors abort before the loop ─────────────────────────────────

def test_start_worker_preflight_failure_aborts(monkeypatch, fake_redis_conn, tmp_path):
    """A preflight backend-import failure must propagate; the worker should
    not enter its task loop with a broken backend."""
    from distributed_validate import worker

    def fail_preflight(mode):
        raise RuntimeError("backend import failed")

    monkeypatch.setattr(worker, "_preflight_backend_import", fail_preflight)

    load_calls = {"n": 0}

    def fake_load(redis_conn, modes, **kwargs):
        load_calls["n"] += 1
        return None, None

    monkeypatch.setattr(worker, "load_next_optimization_task", fake_load)

    with pytest.raises(RuntimeError, match="backend import failed"):
        worker.start_worker(
            redis_host="localhost",
            redis_port=6379,
            supported_modes=["xtb"],
            poll_interval_seconds=0.001,
            log_dir=str(tmp_path / "logs"),
            worker_name="preflight_fail_worker",
        )

    # Loop never started → load was never called
    assert load_calls["n"] == 0


# ── redis unreachable at startup ───────────────────────────────────────────

def test_start_worker_raises_when_redis_unavailable(monkeypatch, tmp_path):
    """Failed ping at startup raises RuntimeError before any loop work."""
    import redis

    from distributed_validate import worker

    class BadRedis:
        def ping(self):
            raise redis.exceptions.RedisError("no server")

    monkeypatch.setattr(redis, "Redis", lambda **kw: BadRedis())

    with pytest.raises(RuntimeError, match="Redis is unavailable"):
        worker.start_worker(
            redis_host="localhost",
            redis_port=6379,
            supported_modes=["xtb"],
            poll_interval_seconds=0.001,
            log_dir=str(tmp_path / "logs"),
            worker_name="no_redis_worker",
        )
