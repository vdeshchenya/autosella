"""Tests for distributed_validate.client.RemoteOptimizationClient.

Covers the full submit → wait lifecycle using fakeredis plus a monkeypatched
``redis`` module so the client builds against a FakeStrictRedis.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_fakeredis


@pytest.fixture
def fake_redis_module(monkeypatch):
    """Patch ``redis.Redis`` to return a FakeStrictRedis.

    The client does ``import redis`` and then ``redis.Redis(host=..., port=...)``.
    We replace only the ``Redis`` constructor inside the real ``redis`` module
    so fakeredis internals (which import from ``redis``) still work.
    """
    import fakeredis
    import redis

    def _fake_redis(host="localhost", port=6379, **kw):
        return fakeredis.FakeStrictRedis()

    monkeypatch.setattr(redis, "Redis", _fake_redis)
    return redis


@pytest.fixture
def client(fake_redis_module):
    from distributed_validate.client import RemoteOptimizationClient

    return RemoteOptimizationClient(
        redis_host="localhost", redis_port=6379, poll_interval_seconds=0.01
    )


def test_client_construction_pings(fake_redis_module):
    """Constructor must ping; FakeStrictRedis supports ping() out of the box."""
    from distributed_validate.client import RemoteOptimizationClient

    RemoteOptimizationClient(redis_host="localhost", redis_port=6379)


def test_client_submit_returns_task_id(client):
    tid = client.submit(
        {"mode": "xtb", "mol_name": "mol_1", "baseline": {}, "max_steps": 10}
    )
    assert tid.startswith("xtb:mol_1:")


def test_client_wait_for_result_success(client):
    """submit → pre-populate a complete result → wait_for_result returns data
    and retains the artifact for reconnect/resume."""
    from distributed_validate.protocol import (
        result_key,
        store_optimization_result,
        task_key,
    )

    tid = client.submit(
        {"mode": "xtb", "mol_name": "m", "baseline": {}, "max_steps": 10}
    )
    payload = {"result": {"rel_steps": 0.5, "rel_energy": 1.0}, "error": None}
    store_optimization_result(client._redis, tid, payload)

    got = client.wait_for_result(tid, timeout_seconds=1.0)
    assert got == payload
    assert client._redis.exists(task_key(tid))
    assert client._redis.exists(result_key(tid))
    assert client.wait_for_result(tid, timeout_seconds=1.0) == payload


def test_client_wait_for_result_failure_raises(client):
    from distributed_validate.protocol import store_optimization_result

    tid = client.submit(
        {"mode": "xtb", "mol_name": "bad", "baseline": {}, "max_steps": 10}
    )
    store_optimization_result(
        client._redis, tid, {"result": None, "error": "backend crashed"}
    )

    with pytest.raises(RuntimeError, match="backend crashed"):
        client.wait_for_result(tid, timeout_seconds=1.0)


def test_client_wait_for_result_timeout(client):
    with pytest.raises(TimeoutError, match="Timed out"):
        client.wait_for_result("never_arrives", timeout_seconds=0.05)


def test_client_raises_when_redis_unavailable(monkeypatch):
    """If ping() fails, constructor should raise RuntimeError."""
    import redis

    class BadRedis:
        def __init__(self, *a, **k):
            pass

        def ping(self):
            raise redis.exceptions.RedisError("no server")

    monkeypatch.setattr(redis, "Redis", lambda **kw: BadRedis())

    from distributed_validate.client import RemoteOptimizationClient

    with pytest.raises(RuntimeError, match="Redis is unavailable"):
        RemoteOptimizationClient(redis_host="localhost", redis_port=6379)


def test_client_raises_when_redis_package_missing(monkeypatch):
    """If redis can't be imported, client construction raises RuntimeError."""
    import builtins
    import sys

    # Remove redis from sys.modules so the next import triggers the import hook
    monkeypatch.delitem(sys.modules, "redis", raising=False)

    real_import = builtins.__import__

    def faux_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("redis not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faux_import)

    from distributed_validate.client import RemoteOptimizationClient

    with pytest.raises(RuntimeError, match="redis package"):
        RemoteOptimizationClient(redis_host="h", redis_port=1)
