"""Task retention, expiry and stale-operation races without numerical work."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from distributed_validate import protocol as p

pytestmark = pytest.mark.requires_fakeredis
OWNED = {"run_id": "run", "evaluation_id": "evaluation"}
RESULT = {"result": {"mol_name": "m"}, "error": None}


@pytest.fixture
def redis_conn():
    import fakeredis
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def clock(monkeypatch):
    # Advance the Redis test double's wall clock, not the test runner's monotonic
    # deadline. This exercises a full 12-hour lifetime without sleeping for it.
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


def submit(redis_conn):
    return p.submit_optimization_task(
        redis_conn, {"mode": "xtb", "mol_name": "m", **OWNED}, task_id="task",
    )


def keys(task_id, *, result=False):
    items = [p.task_key(task_id), p.task_state_key(task_id)]
    return items + [p.result_key(task_id)] if result else items


def assert_retention(redis_conn, task_id, expected, *, result=False):
    assert [redis_conn.ttl(key) for key in keys(task_id, result=result)] == [expected] * (
        3 if result else 2
    )


def test_submission_expires_after_twelve_hours_and_stale_queue_entry_is_skipped(redis_conn, clock):
    assert p.TASK_TTL_SECONDS == 43200
    task_id = submit(redis_conn)
    assert_retention(redis_conn, task_id, 43200)
    clock[0] += 43199
    assert submit(redis_conn) == task_id
    assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "queued"
    assert_retention(redis_conn, task_id, 1)
    clock[0] += 2
    assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "missing"
    assert redis_conn.llen(p.queue_key("xtb")) == 1
    assert p.load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker") == (None, None)
    assert redis_conn.llen(p.queue_key("xtb")) == 0
    assert not redis_conn.exists(*keys(task_id), p.worker_key("worker"))


def test_claim_and_owned_heartbeats_preserve_a_long_running_task(redis_conn, clock):
    task_id = submit(redis_conn)
    clock[0] += 43199
    assert p.load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")[0] == task_id
    assert_retention(redis_conn, task_id, 43200)
    for _ in range(3):
        clock[0] += 21600
        p.heartbeat_worker(redis_conn, "worker", task_id)
        assert_retention(redis_conn, task_id, 43200)
        assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "running"
    clock[0] += 43201
    p.heartbeat_worker(redis_conn, "worker", task_id)
    assert not redis_conn.exists(*keys(task_id))
    assert redis_conn.get(p.worker_key("worker")) == b""


@pytest.mark.parametrize("missing", ["input", "state"])
def test_heartbeat_does_not_recreate_partial_task_records(redis_conn, clock, missing):
    task_id = submit(redis_conn)
    p.load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")
    clock[0] += 100
    gone, retained = keys(task_id) if missing == "input" else keys(task_id)[::-1]
    redis_conn.delete(gone)
    p.heartbeat_worker(redis_conn, "worker", task_id)
    assert not redis_conn.exists(gone)
    assert redis_conn.ttl(retained) == 43100


def test_unowned_heartbeat_cannot_extend_task_retention(redis_conn, clock):
    task_id = submit(redis_conn)
    p.load_next_optimization_task(redis_conn, ["xtb"], worker_id="owner")
    clock[0] += 100
    p.heartbeat_worker(redis_conn, "other", task_id)
    assert_retention(redis_conn, task_id, 43100)
    assert redis_conn.hget(p.task_state_key(task_id), "worker_id") == b"owner"
    assert redis_conn.get(p.worker_key("other")) == b""


def test_pause_and_resume_refresh_retention_but_cannot_resume_expired_input(redis_conn, clock):
    task_id = submit(redis_conn)
    clock[0] += 43199
    assert p.withdraw_queued_task(redis_conn, task_id, **OWNED)
    assert_retention(redis_conn, task_id, 43200)
    clock[0] += 43199
    assert p.resume_queued_task(redis_conn, task_id, **OWNED)
    assert_retention(redis_conn, task_id, 43200)
    assert redis_conn.lrange(p.queue_key("xtb"), 0, -1) == [task_id.encode()]
    assert p.withdraw_queued_task(redis_conn, task_id, **OWNED)
    redis_conn.delete(p.task_key(task_id))
    clock[0] += 100
    assert not p.resume_queued_task(redis_conn, task_id, **OWNED)
    assert not redis_conn.exists(p.task_key(task_id))
    assert redis_conn.ttl(p.task_state_key(task_id)) == 43100
    assert redis_conn.llen(p.queue_key("xtb")) == 0
    clock[0] += 43200
    assert not p.resume_queued_task(redis_conn, task_id, **OWNED)
    assert not redis_conn.exists(*keys(task_id))


def test_lost_fence_has_finite_retention_and_status_reads_do_not_extend_it(redis_conn, clock):
    task_id = submit(redis_conn)
    clock[0] += 43199
    redis_conn.lrem(p.queue_key("xtb"), 0, task_id)
    assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "lost"
    assert_retention(redis_conn, task_id, 43200)
    clock[0] += 100
    assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "lost"
    assert_retention(redis_conn, task_id, 43100)
    clock[0] += 43200
    assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "missing"


@pytest.mark.parametrize("payload", [RESULT, {"result": None, "error": "optimizer failed"}])
def test_completion_retains_all_records_then_expires_without_polling_extensions(redis_conn, clock, payload):
    task_id = submit(redis_conn)
    p.load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")
    clock[0] += 43199
    p.store_optimization_result(redis_conn, task_id, payload)
    p.finish_worker_task(redis_conn, "worker", task_id)
    assert_retention(redis_conn, task_id, 43200, result=True)
    clock[0] += 100
    assert submit(redis_conn) == task_id
    p.store_optimization_result(redis_conn, task_id, {"error": "late duplicate"})
    p.heartbeat_worker(redis_conn, "worker", task_id)
    assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "finished"
    assert p.get_optimization_result(redis_conn, task_id)["data"] == payload
    assert_retention(redis_conn, task_id, 43100, result=True)
    clock[0] += 43101
    p.heartbeat_worker(redis_conn, "worker", task_id)
    assert p.get_optimization_result(redis_conn, task_id) is None
    assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "missing"
    assert not redis_conn.exists(*keys(task_id, result=True))


def test_late_result_has_bounded_retention_without_reconstructing_expired_input(redis_conn, clock):
    task_id = submit(redis_conn)
    clock[0] += 43201
    p.store_optimization_result(redis_conn, task_id, RESULT)
    assert not redis_conn.exists(p.task_key(task_id))
    assert redis_conn.ttl(p.task_state_key(task_id)) == 43200
    assert redis_conn.ttl(p.result_key(task_id)) == 43200
    assert p.get_optimization_result(redis_conn, task_id)["data"] == RESULT


@pytest.mark.parametrize("interruption", ["delete", "finish"])
def test_heartbeat_race_does_not_recreate_or_renew_terminal_state(redis_conn, clock, monkeypatch, interruption):
    task_id = submit(redis_conn)
    p.load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")
    original_pipeline = redis_conn.pipeline
    interrupted = []

    def pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        execute = pipe.execute

        def raced_execute(*args, **kwargs):
            if not interrupted:
                interrupted.append(True)
                if interruption == "delete":
                    redis_conn.delete(*keys(task_id))
                else:
                    p.store_optimization_result(redis_conn, task_id, RESULT)
                    clock[0] += 100
            return execute(*args, **kwargs)

        pipe.execute = raced_execute
        return pipe

    monkeypatch.setattr(redis_conn, "pipeline", pipeline)
    p.heartbeat_worker(redis_conn, "worker", task_id)
    if interruption == "delete":
        assert not redis_conn.exists(*keys(task_id))
    else:
        assert p.inspect_optimization_task(redis_conn, task_id)["state"] == "finished"
        assert_retention(redis_conn, task_id, 43100, result=True)


@pytest.fixture
def isolated_redis():
    """A private Redis process with no TCP listener or persisted production data."""
    import redis

    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is not installed")
    # Keep the Unix socket below the platform's path-length limit.
    with tempfile.TemporaryDirectory(prefix="task-ttl-", dir="/tmp") as directory:
        socket = str(Path(directory) / "redis.sock")
        process = subprocess.Popen(
            [executable, "--port", "0", "--unixsocket", socket,
             "--unixsocketperm", "700", "--save", "", "--appendonly", "no",
             "--dir", directory, "--loglevel", "warning"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        client = redis.Redis(unix_socket_path=socket, socket_timeout=1)
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    client.ping()
                    break
                except redis.exceptions.ConnectionError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        pytest.fail("private Redis did not start")
                    time.sleep(0.01)
            yield client
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()


def test_real_redis_expiration_and_late_heartbeat_do_not_recreate_records(isolated_redis):
    task_id = submit(isolated_redis)
    p.load_next_optimization_task(isolated_redis, ["xtb"], worker_id="worker")
    p.store_optimization_result(isolated_redis, task_id, RESULT)
    assert all(43198 <= isolated_redis.ttl(key) <= 43200 for key in keys(task_id, result=True))
    # Verify the real server's expiry semantics at a shortened deadline after
    # checking that production writes actually installed the 12-hour lifetime.
    for key in keys(task_id, result=True):
        isolated_redis.pexpire(key, 1)
    deadline = time.monotonic() + 2
    while isolated_redis.exists(*keys(task_id, result=True)):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    p.heartbeat_worker(isolated_redis, "worker", task_id)
    assert p.inspect_optimization_task(isolated_redis, task_id)["state"] == "missing"
    assert not isolated_redis.exists(*keys(task_id, result=True))


def test_real_redis_expiry_during_heartbeat_transaction_does_not_recreate_state(isolated_redis, monkeypatch):
    task_id = submit(isolated_redis)
    p.load_next_optimization_task(isolated_redis, ["xtb"], worker_id="worker")
    # Set the deadline before WATCH, so expiry itself must invalidate the
    # heartbeat transaction, rather than an intervening PEXPIRE command.
    for key in keys(task_id):
        isolated_redis.pexpire(key, 200)
    original_pipeline = isolated_redis.pipeline
    expired = []

    def pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        execute = pipe.execute

        def raced_execute(*args, **kwargs):
            if not expired:
                expired.append(True)
                assert isolated_redis.exists(*keys(task_id)) == 2
                deadline = time.monotonic() + 2
                while isolated_redis.exists(*keys(task_id)):
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
            return execute(*args, **kwargs)

        pipe.execute = raced_execute
        return pipe

    monkeypatch.setattr(isolated_redis, "pipeline", pipeline)
    p.heartbeat_worker(isolated_redis, "worker", task_id)
    assert not isolated_redis.exists(*keys(task_id))
