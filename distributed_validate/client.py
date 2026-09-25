from __future__ import annotations

import time

from distributed_validate.protocol import (
    get_optimization_result,
    inspect_optimization_task,
    resume_queued_task,
    submit_optimization_task,
    withdraw_queued_task,
)


class InfrastructureUnavailable(RuntimeError):
    """The transport failed; this says nothing about candidate quality."""


class RemoteOptimizationClient:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        poll_interval_seconds: float = 0.1,
        redis_db: int = 0,
    ):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "redis package is required for distributed validation"
            ) from exc

        self._redis = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db,
            socket_connect_timeout=5.0, socket_timeout=5.0,
            retry=redis.retry.Retry(redis.backoff.NoBackoff(), 0),
        )
        try:
            self._redis.ping()
        except redis.exceptions.RedisError as exc:
            raise InfrastructureUnavailable(
                f"Redis is unavailable at {redis_host}:{redis_port}"
            ) from exc
        self.poll_interval_seconds = poll_interval_seconds

    def submit(self, task: dict, task_id: str | None = None) -> str:
        return submit_optimization_task(self._redis, task, task_id=task_id)

    def reconnect(self) -> None:
        self._redis.connection_pool.disconnect()
        self._redis.ping()

    def get_result(self, task_id: str) -> dict | None:
        payload = get_optimization_result(self._redis, task_id)
        return None if payload is None else payload["data"]

    def task_status(self, task_id: str) -> dict:
        return inspect_optimization_task(self._redis, task_id)

    def withdraw_queued_task(self, task_id: str, *, run_id: str, evaluation_id: str) -> bool:
        return withdraw_queued_task(self._redis, task_id, run_id=run_id,
                                    evaluation_id=evaluation_id)

    def resume_queued_task(self, task_id: str, *, run_id: str, evaluation_id: str) -> bool:
        return resume_queued_task(self._redis, task_id, run_id=run_id,
                                  evaluation_id=evaluation_id)

    def wait_for_result(self, task_id: str, timeout_seconds: float) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while True:
            payload = get_optimization_result(self._redis, task_id)
            if payload is not None:
                result = payload["data"]
                if payload["status"] != "complete":
                    raise RuntimeError(
                        f"Worker failed for task {task_id}: "
                        f"{result.get('error', 'unknown error')}"
                    )
                return result

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for distributed result for {task_id}")

            time.sleep(self.poll_interval_seconds)


# Backward-compatible alias for any existing local imports.
RemoteComputeClient = RemoteOptimizationClient
