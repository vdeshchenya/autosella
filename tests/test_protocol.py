"""Tests for distributed_validate.protocol — uses fakeredis."""

from __future__ import annotations


def test_prepared_protocol_cannot_mix_with_legacy_queues():
    from distributed_validate.protocol import queue_key, task_key
    assert queue_key("xtb") == "validate:prepared-v1:queue:xtb"
    assert task_key("task") == "validate:prepared-v1:taskdata:task"

import re

import pytest

pytestmark = pytest.mark.requires_fakeredis


@pytest.fixture
def redis_conn():
    import fakeredis
    return fakeredis.FakeStrictRedis()


def test_create_task_id_format():
    from distributed_validate.protocol import create_task_id
    tid = create_task_id("xtb", "386403613_13")
    # Expected: xtb:386403613_13:<8 hex>
    assert re.match(r"^xtb:386403613_13:[0-9a-f]{8}$", tid)


def test_create_task_id_escapes_colons():
    """mol_name with colon (e.g. 'mol:name') gets underscored."""
    from distributed_validate.protocol import create_task_id
    tid = create_task_id("ff", "foo:bar")
    assert "foo_bar" in tid


def test_submit_and_load_round_trip(redis_conn):
    from distributed_validate.protocol import (
        submit_optimization_task,
        load_next_optimization_task,
    )
    task = {
        "mode": "xtb",
        "mol_name": "test_mol",
        "baseline": {"n_steps": 10, "improvement": 100.0, "n_atoms": 5, "xyz_path": "/tmp/foo.xyz"},
        "max_steps": 50,
    }
    task_id = submit_optimization_task(redis_conn, task)
    assert task_id.startswith("xtb:test_mol:")

    loaded_id, loaded_task = load_next_optimization_task(redis_conn, ["xtb"])
    assert loaded_id == task_id
    assert loaded_task["mode"] == "xtb"
    assert loaded_task["mol_name"] == "test_mol"
    assert loaded_task["baseline"]["n_atoms"] == 5
    assert loaded_task["task_id"] == task_id


def test_load_no_tasks_returns_none(redis_conn):
    from distributed_validate.protocol import load_next_optimization_task
    tid, t = load_next_optimization_task(redis_conn, ["xtb"])
    assert tid is None
    assert t is None


def test_store_and_get_result_success(redis_conn):
    from distributed_validate.protocol import (
        get_optimization_result,
        store_optimization_result,
    )
    payload = {"result": {"rel_steps": 0.5, "rel_energy": 1.0}, "error": None}
    store_optimization_result(redis_conn, "task_x", payload)
    got = get_optimization_result(redis_conn, "task_x")
    assert got["status"] == "complete"
    assert got["data"] == payload


def test_store_and_get_result_failed(redis_conn):
    from distributed_validate.protocol import (
        get_optimization_result,
        store_optimization_result,
    )
    payload = {"result": None, "error": "something broke"}
    store_optimization_result(redis_conn, "task_y", payload)
    got = get_optimization_result(redis_conn, "task_y")
    assert got["status"] == "failed"
    assert got["data"]["error"] == "something broke"


def test_get_missing_result_returns_none(redis_conn):
    from distributed_validate.protocol import get_optimization_result
    assert get_optimization_result(redis_conn, "nonexistent") is None


def test_delete_task_artifacts_cleans_up(redis_conn):
    from distributed_validate.protocol import (
        delete_task_artifacts,
        get_optimization_result,
        result_key,
        store_optimization_result,
        submit_optimization_task,
        task_key,
    )
    task_id = submit_optimization_task(
        redis_conn,
        {"mode": "xtb", "mol_name": "foo", "baseline": {}, "max_steps": 10},
    )
    store_optimization_result(redis_conn, task_id, {"result": {}, "error": None})
    delete_task_artifacts(redis_conn, task_id)

    assert not redis_conn.exists(task_key(task_id))
    assert not redis_conn.exists(result_key(task_id))
    assert get_optimization_result(redis_conn, task_id) is None


def test_idempotent_replay_does_not_duplicate_queued_or_running_task(redis_conn):
    from distributed_validate.protocol import (
        inspect_optimization_task, load_next_optimization_task, queue_key,
        submit_optimization_task,
    )
    task = {"mode": "xtb", "mol_name": "m"}
    tid = submit_optimization_task(redis_conn, task, task_id="stable-id")
    assert submit_optimization_task(redis_conn, task, task_id=tid) == tid
    assert redis_conn.llen(queue_key("xtb")) == 1
    claimed, _ = load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")
    assert claimed == tid
    assert inspect_optimization_task(redis_conn, tid)["state"] == "running"
    assert submit_optimization_task(redis_conn, task, task_id=tid) == tid
    assert redis_conn.llen(queue_key("xtb")) == 0


def test_results_are_immutable_and_have_bounded_retention(redis_conn):
    from distributed_validate.protocol import (
        TASK_TTL_SECONDS, get_optimization_result, result_key, store_optimization_result,
    )
    payload = {"result": {"mol_name": "m"}, "error": None}
    store_optimization_result(redis_conn, "id", payload)
    store_optimization_result(redis_conn, "id", {"result": None, "error": "late duplicate"})
    assert get_optimization_result(redis_conn, "id")["data"] == payload
    assert get_optimization_result(redis_conn, "id")["data"] == payload
    assert 0 < redis_conn.ttl(result_key("id")) <= TASK_TTL_SECONDS


def test_heartbeat_and_lost_queue_evidence(redis_conn):
    from distributed_validate.protocol import (
        heartbeat_worker, inspect_optimization_task, load_next_optimization_task,
        queue_key, submit_optimization_task, worker_key,
    )
    tid = submit_optimization_task(redis_conn, {"mode": "xtb", "mol_name": "m"})
    assert inspect_optimization_task(redis_conn, tid)["state"] == "queued"
    load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")
    redis_conn.delete(worker_key("worker"))
    assert inspect_optimization_task(redis_conn, tid)["state"] == "lease_expired"
    heartbeat_worker(redis_conn, "worker", tid)
    assert inspect_optimization_task(redis_conn, tid)["state"] == "running"
    queued = submit_optimization_task(redis_conn, {"mode": "xtb", "mol_name": "other"})
    redis_conn.lrem(queue_key("xtb"), 0, queued)
    assert inspect_optimization_task(redis_conn, queued)["state"] == "lost"


def test_withdraw_checks_ownership_and_does_not_interrupt_running_tasks(redis_conn):
    from distributed_validate.protocol import (
        load_next_optimization_task, resume_queued_task, submit_optimization_task,
        withdraw_queued_task,
    )
    owned = {"run_id": "run", "evaluation_id": "eval"}
    tid = submit_optimization_task(redis_conn, {"mode": "xtb", "mol_name": "m", **owned})
    assert not withdraw_queued_task(redis_conn, tid, run_id="other", evaluation_id="eval")
    assert withdraw_queued_task(redis_conn, tid, **owned)
    assert load_next_optimization_task(redis_conn, ["xtb"]) == (None, None)
    assert resume_queued_task(redis_conn, tid, **owned)
    load_next_optimization_task(redis_conn, ["xtb"], worker_id="alive")
    assert not withdraw_queued_task(redis_conn, tid, **owned)


def test_claim_contention_yields_without_losing_task(redis_conn, monkeypatch):
    from redis.exceptions import WatchError
    from distributed_validate.protocol import (
        load_next_optimization_task, queue_key, submit_optimization_task,
        task_state_key, worker_key,
    )

    submit_optimization_task(redis_conn, {"mode": "xtb", "mol_name": "m"}, task_id="contended")
    original_pipeline = redis_conn.pipeline
    conflicts = []

    def contended_pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)

        def execute(*args, **kwargs):
            conflicts.append(True)
            pipe.reset()
            raise WatchError("Watched variable changed.")

        pipe.execute = execute
        return pipe

    monkeypatch.setattr(redis_conn, "pipeline", contended_pipeline)
    assert load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker") == (None, None)
    assert len(conflicts) == 32
    assert redis_conn.lrange(queue_key("xtb"), 0, -1) == [b"contended"]
    assert redis_conn.hget(task_state_key("contended"), "state") == b"queued"
    assert not redis_conn.exists(worker_key("worker"))

    monkeypatch.setattr(redis_conn, "pipeline", original_pipeline)
    task_id, task = load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")
    assert task_id == "contended" and task["mol_name"] == "m"
    assert redis_conn.llen(queue_key("xtb")) == 0
    assert redis_conn.hget(task_state_key(task_id), "state") == b"running"


def test_claim_connection_failure_is_not_treated_as_contention(redis_conn, monkeypatch):
    from redis.exceptions import WatchError
    from distributed_validate.protocol import load_next_optimization_task, submit_optimization_task

    submit_optimization_task(redis_conn, {"mode": "xtb", "mol_name": "m"}, task_id="connection")
    original_pipeline = redis_conn.pipeline

    def broken_pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)

        def execute(*args, **kwargs):
            pipe.reset()
            raise WatchError("A ConnectionError occurred while watching one or more keys")

        pipe.execute = execute
        return pipe

    monkeypatch.setattr(redis_conn, "pipeline", broken_pipeline)
    with pytest.raises(WatchError, match="ConnectionError"):
        load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker")


def test_watched_connection_failure_escapes_for_bounded_recovery(redis_conn, monkeypatch):
    from redis.exceptions import WatchError
    from distributed_validate.protocol import queue_key, submit_optimization_task

    original_pipeline = redis_conn.pipeline
    failed = []

    def broken_acknowledgement(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        original_execute = pipe.execute

        def execute(*args, **kwargs):
            result = original_execute(*args, **kwargs)
            if not failed:
                failed.append(True)
                raise WatchError("A ConnectionError occurred while watching one or more keys")
            return result

        pipe.execute = execute
        return pipe

    monkeypatch.setattr(redis_conn, "pipeline", broken_acknowledgement)
    task = {"mode": "xtb", "mol_name": "m"}
    with pytest.raises(WatchError, match="ConnectionError"):
        submit_optimization_task(redis_conn, task, task_id="durable")
    assert submit_optimization_task(redis_conn, task, task_id="durable") == "durable"
    assert redis_conn.llen(queue_key("xtb")) == 1


def test_withdraw_resume_race_cannot_report_queued_task_as_lost(redis_conn, monkeypatch):
    from distributed_validate.protocol import (
        inspect_optimization_task, queue_key, resume_queued_task,
        submit_optimization_task, task_state_key, withdraw_queued_task,
    )
    owned = {"run_id": "run", "evaluation_id": "eval"}
    tid = submit_optimization_task(redis_conn, {"mode": "xtb", "mol_name": "m", **owned})
    stages = []
    original_pipeline = redis_conn.pipeline

    def instrument(transport):
        original_hgetall, original_lpos = transport.hgetall, transport.lpos

        def hgetall(key):
            value = original_hgetall(key)
            if key == task_state_key(tid) and not stages:
                stages.append("withdraw")
                assert withdraw_queued_task(redis_conn, tid, **owned)
            return value

        def lpos(key, value):
            position = original_lpos(key, value)
            if key == queue_key("xtb") and stages == ["withdraw"]:
                stages.append("resume")
                assert position is None
                assert resume_queued_task(redis_conn, tid, **owned)
            return position

        monkeypatch.setattr(transport, "hgetall", hgetall)
        monkeypatch.setattr(transport, "lpos", lpos)
        return transport

    instrument(redis_conn)
    monkeypatch.setattr(redis_conn, "pipeline", lambda *a, **kw: instrument(original_pipeline(*a, **kw)))
    assert inspect_optimization_task(redis_conn, tid)["state"] == "queued"
    assert stages == ["withdraw", "resume"]
    assert redis_conn.llen(queue_key("xtb")) == 1


def test_confirmed_lost_task_is_fenced_against_late_resume_or_stale_queue_entry(redis_conn):
    from distributed_validate.protocol import (
        inspect_optimization_task, load_next_optimization_task, queue_key,
        resume_queued_task, submit_optimization_task, withdraw_queued_task,
    )
    owned = {"run_id": "run", "evaluation_id": "eval"}
    tid = submit_optimization_task(redis_conn, {"mode": "xtb", "mol_name": "m", **owned})
    redis_conn.lrem(queue_key("xtb"), 0, tid)
    assert inspect_optimization_task(redis_conn, tid)["state"] == "lost"
    assert not withdraw_queued_task(redis_conn, tid, **owned)
    assert not resume_queued_task(redis_conn, tid, **owned)
    redis_conn.lpush(queue_key("xtb"), tid)
    assert load_next_optimization_task(redis_conn, ["xtb"], worker_id="worker") == (None, None)
