from __future__ import annotations

import pickle
import time
import uuid
from typing import Iterable

# Prepared-source tasks require leases, retained results, and the new stopping
# contract. Legacy workers must never consume them during a rolling migration.
QUEUE_PREFIX = "validate:prepared-v1"
# Retain task records for 12 hours after their last lifecycle transition.
# Owned running heartbeats renew inputs/state; terminal reads do not renew them.
TASK_TTL_SECONDS = 12 * 60 * 60


def _check_watch_retry(exc, conflicts: int) -> None:
    """Redis wraps broken watched connections in WatchError; do not spin on it."""
    from redis.exceptions import RedisError, WatchError

    context = exc.__cause__ or exc.__context__
    if isinstance(context, RedisError) and not isinstance(context, WatchError):
        raise context from exc
    if "occurred while watching" in str(exc) or conflicts >= 32:
        raise exc


def queue_key(mode: str) -> str:
    return f"{QUEUE_PREFIX}:queue:{mode}"


def task_key(task_id: str) -> str:
    return f"{QUEUE_PREFIX}:taskdata:{task_id}"


def result_key(task_id: str) -> str:
    return f"{QUEUE_PREFIX}:result:{task_id}"


def task_state_key(task_id: str) -> str:
    return f"{QUEUE_PREFIX}:taskstate:{task_id}"


def worker_key(worker_id: str) -> str:
    return f"{QUEUE_PREFIX}:worker:{worker_id}"


def _refresh_task_retention(pipe, task_id: str, *, result: bool = False) -> None:
    """Queue expirations in the same transaction as a lifecycle write.

    EXPIRE only touches existing keys, so missing input is never reconstructed.
    """
    pipe.expire(task_key(task_id), TASK_TTL_SECONDS)
    pipe.expire(task_state_key(task_id), TASK_TTL_SECONDS)
    if result:
        pipe.expire(result_key(task_id), TASK_TTL_SECONDS)


def create_task_id(mode: str, mol_name: str) -> str:
    return (
        f"{mode}:{mol_name.replace(':', '_')}:"
        f"{uuid.uuid4().hex[:8]}"
    )


def submit_optimization_task(redis_conn, task: dict, task_id: str | None = None) -> str:
    """Atomically submit once, including after a lost Redis acknowledgement.

    Recovery callers must persist the explicit ID *before* making this call.
    Task records have bounded retention; replaying an existing ID does not
    refresh its TTL. Recovery must diagnose expired submitted IDs, not rerun them.
    """
    from redis.exceptions import WatchError

    task_id = task_id or task.get("task_id") or create_task_id(
        mode=task["mode"],
        mol_name=task["mol_name"],
    )
    payload = dict(task)
    payload["task_id"] = task_id
    with redis_conn.pipeline() as pipe:
        conflicts = 0
        while True:
            try:
                pipe.watch(task_key(task_id), task_state_key(task_id), result_key(task_id))
                if pipe.exists(task_key(task_id), task_state_key(task_id), result_key(task_id)):
                    return task_id
                pipe.multi()
                pipe.set(task_key(task_id), pickle.dumps(payload))
                pipe.hset(task_state_key(task_id), mapping={
                    "state": "queued", "mode": task["mode"],
                    "submitted_at": str(time.time()),
                })
                _refresh_task_retention(pipe, task_id)
                pipe.lpush(queue_key(task["mode"]), task_id)
                pipe.execute()
                return task_id
            except WatchError as exc:
                conflicts += 1
                _check_watch_retry(exc, conflicts)
                continue


def load_next_optimization_task(
    redis_conn, modes: Iterable[str], worker_id: str | None = None,
    lease_seconds: float = 30.0,
) -> tuple[str | None, dict | None]:
    from redis.exceptions import WatchError

    for mode in modes:
        with redis_conn.pipeline() as pipe:
            conflicts = 0
            while True:
                try:
                    pipe.watch(queue_key(mode))
                    task_id = pipe.lindex(queue_key(mode), -1)
                    if not task_id:
                        break
                    task_id_str = task_id.decode()
                    pipe.watch(task_key(task_id_str), task_state_key(task_id_str), result_key(task_id_str))
                    task_data = pipe.get(task_key(task_id_str))
                    state = pipe.hget(task_state_key(task_id_str), "state")
                    skip = task_data is None or pipe.exists(result_key(task_id_str)) or state in (
                        b"running", b"withdrawn", b"lost", b"finished",
                    )
                    pipe.multi()
                    pipe.rpop(queue_key(mode))
                    if not skip:
                        pipe.hset(task_state_key(task_id_str), mapping={
                            "state": "running", "mode": mode,
                            "worker_id": worker_id or "",
                            "claimed_at": str(time.time()),
                            "lease_until": str(time.time() + lease_seconds),
                        })
                        _refresh_task_retention(pipe, task_id_str)
                        if worker_id:
                            pipe.set(worker_key(worker_id), task_id_str, px=max(1, int(lease_seconds * 1000)))
                    pipe.execute()
                    if not skip:
                        return task_id_str, pickle.loads(task_data)
                except WatchError as exc:
                    conflicts += 1
                    # A lost claim under contention has not consumed a task.
                    # Yield to the worker's poll delay after a bounded batch;
                    # connection failures still escape for recovery.
                    _check_watch_retry(exc, 0)
                    if conflicts >= 32:
                        return None, None
                    continue
    return None, None


def heartbeat_worker(redis_conn, worker_id: str, task_id: str | None = None,
                     lease_seconds: float = 30.0) -> None:
    from redis.exceptions import WatchError

    with redis_conn.pipeline() as pipe:
        conflicts = 0
        while True:
            try:
                active = False
                if task_id:
                    pipe.watch(task_key(task_id), task_state_key(task_id), result_key(task_id))
                    active = (
                        pipe.exists(task_key(task_id))
                        and not pipe.exists(result_key(task_id))
                        and pipe.hget(task_state_key(task_id), "state") == b"running"
                        and pipe.hget(task_state_key(task_id), "worker_id") == worker_id.encode()
                    )
                pipe.multi()
                pipe.set(worker_key(worker_id), task_id if active else "",
                         px=max(1, int(lease_seconds * 1000)))
                if active:
                    pipe.hset(task_state_key(task_id), mapping={
                        "lease_until": str(time.time() + lease_seconds),
                    })
                    _refresh_task_retention(pipe, task_id)
                pipe.execute()
                return
            except WatchError as exc:
                conflicts += 1
                _check_watch_retry(exc, conflicts)


def finish_worker_task(redis_conn, worker_id: str, task_id: str) -> None:
    # Result publication precedes this call; preserve its terminal task state.
    redis_conn.set(worker_key(worker_id), "", ex=30)


def _change_queued_task(redis_conn, task_id: str, *, run_id: str,
                        evaluation_id: str, resume: bool) -> bool:
    from redis.exceptions import WatchError

    with redis_conn.pipeline() as pipe:
        conflicts = 0
        while True:
            try:
                pipe.watch(task_key(task_id), task_state_key(task_id), result_key(task_id))
                raw = pipe.get(task_key(task_id))
                if raw is None or pipe.exists(result_key(task_id)):
                    return False
                task = pickle.loads(raw)
                if task.get("run_id") != run_id or task.get("evaluation_id") != evaluation_id:
                    return False
                before, after = ("withdrawn", "queued") if resume else ("queued", "withdrawn")
                if pipe.hget(task_state_key(task_id), "state") != before.encode():
                    return False
                pipe.multi()
                pipe.hset(task_state_key(task_id), mapping={"state": after})
                _refresh_task_retention(pipe, task_id)
                if resume:
                    pipe.lpush(queue_key(task["mode"]), task_id)
                else:
                    pipe.lrem(queue_key(task["mode"]), 0, task_id)
                pipe.execute()
                return True
            except WatchError as exc:
                conflicts += 1
                _check_watch_retry(exc, conflicts)
                continue


def withdraw_queued_task(redis_conn, task_id: str, *, run_id: str,
                         evaluation_id: str) -> bool:
    """Withdraw only an owned queued task; running work and results are retained."""
    return _change_queued_task(redis_conn, task_id, run_id=run_id,
                               evaluation_id=evaluation_id, resume=False)


def resume_queued_task(redis_conn, task_id: str, *, run_id: str,
                       evaluation_id: str) -> bool:
    return _change_queued_task(redis_conn, task_id, run_id=run_id,
                               evaluation_id=evaluation_id, resume=True)


def inspect_optimization_task(redis_conn, task_id: str) -> dict:
    """Inspect a consistent snapshot and fence a confirmed lost queued task.

    Fencing prevents withdrawal/resume from restoring an old ID after recovery
    has observed it lost and prepared a replacement. Expired heartbeats remain
    ambiguous; they never fence or resubmit a running optimization.
    """
    from redis.exceptions import WatchError

    with redis_conn.pipeline() as pipe:
        conflicts = 0
        while True:
            try:
                pipe.watch(result_key(task_id), task_state_key(task_id), task_key(task_id))
                fence_lost = False
                if pipe.exists(result_key(task_id)):
                    state = {"state": "finished"}
                else:
                    raw = pipe.hgetall(task_state_key(task_id))
                    state = {key.decode(): value.decode() for key, value in raw.items()}
                    if not state:
                        state = {"state": "unknown" if pipe.exists(task_key(task_id)) else "missing"}
                    elif state.get("state") == "queued":
                        key = queue_key(state["mode"])
                        pipe.watch(key)
                        if pipe.lpos(key, task_id) is None:
                            state["state"] = "lost"
                            state["lost_at"] = str(time.time())
                            fence_lost = True
                    elif state.get("state") == "running":
                        worker_id = state.get("worker_id")
                        if not worker_id:
                            state["state"] = "unknown"
                        else:
                            pipe.watch(worker_key(worker_id))
                            if pipe.get(worker_key(worker_id)) != task_id.encode():
                                state["state"] = "lease_expired"
                pipe.multi()
                if fence_lost:
                    pipe.hset(task_state_key(task_id), mapping={
                        "state": "lost", "lost_at": state["lost_at"],
                    })
                    _refresh_task_retention(pipe, task_id)
                pipe.execute()
                return state
            except WatchError as exc:
                conflicts += 1
                _check_watch_retry(exc, conflicts)


def store_optimization_result(redis_conn, task_id: str, result: dict) -> None:
    from redis.exceptions import WatchError

    status = "failed" if result.get("error") else "complete"
    with redis_conn.pipeline() as pipe:
        conflicts = 0
        while True:
            try:
                pipe.watch(result_key(task_id), task_state_key(task_id), task_key(task_id))
                if pipe.exists(result_key(task_id)):
                    return
                pipe.multi()
                pipe.hset(result_key(task_id), mapping={
                    "status": status, "data": pickle.dumps(result),
                    "timestamp": str(time.time()),
                })
                pipe.hset(task_state_key(task_id), mapping={"state": "finished"})
                _refresh_task_retention(pipe, task_id, result=True)
                pipe.execute()
                return
            except WatchError as exc:
                conflicts += 1
                _check_watch_retry(exc, conflicts)
                continue


def get_optimization_result(redis_conn, task_id: str) -> dict | None:
    key = result_key(task_id)
    payload = redis_conn.hgetall(key)
    status = payload.get(b"status")
    data = payload.get(b"data")
    if status is None or data is None:
        return None
    return {
        "status": status.decode(),
        "data": pickle.loads(data),
    }


def delete_task_artifacts(redis_conn, task_id: str) -> None:
    redis_conn.delete(result_key(task_id))
    redis_conn.delete(task_key(task_id))
    redis_conn.delete(task_state_key(task_id))


# Backward-compatible aliases for older imports in this repo.
submit_task = submit_optimization_task
load_next_task = load_next_optimization_task
store_result = store_optimization_result
get_result = get_optimization_result
