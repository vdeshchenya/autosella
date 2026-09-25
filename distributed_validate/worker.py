from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import logging
import math
import multiprocessing as mp
import operator
import os
import queue
import signal
import sys
import time
import traceback
import tempfile
import uuid
from datetime import datetime

# Cap in-process threading libraries BEFORE numpy/xTB pull libgomp /
# openblas / MKL into the address space. Pool sizes and the wait policy
# are baked in at library load time; once loaded, OMP_NUM_THREADS env
# changes are ignored and only a runtime ctypes hack can resize them.
# OMP_WAIT_POLICY=PASSIVE / GOMP_SPINCOUNT=0 prevent idle OpenMP threads
# from busy-waiting and burning CPU.
for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OMP_WAIT_POLICY", "PASSIVE"),
    ("OMP_DYNAMIC", "FALSE"),
    ("MKL_DYNAMIC", "FALSE"),
    ("GOMP_SPINCOUNT", "0"),
):
    os.environ.setdefault(_var, _val)

# Force JAX to enable x64. Algorithms in algo.py request float64 explicitly
# (e.g. jnp.asarray(..., dtype=np.float64)); without this JAX silently
# downcasts to float32, which corrupts the optimizer's numerics.
# Set unconditionally (overwriting any user value) so workers always run
# in the float64 regime regardless of the launching shell.
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

from convergence import init_convergence_state, is_converged, update_convergence_state
from distributed_validate.diagnostics import TaskCapture, attach_bundle
from distributed_validate.optimizer import (
    describe_optimizer_spec,
    load_minimize_func,
    optimizer_cache_key,
)
from distributed_validate.protocol import (
    finish_worker_task,
    heartbeat_worker,
    load_next_optimization_task,
    store_optimization_result,
)


def str_to_bool(value):
    if value.lower() in ("true", "yes", "1"):
        return True
    elif value.lower() in ("false", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def setup_logging(component_name, verbose=False, log_dir="logs", log_file=None):
    """Configure logging with file handler."""
    logger = logging.getLogger(component_name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{component_name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger


def get_time():
    return datetime.now().strftime("%H:%M:%S")


# Exact by definition (thermochemical calorie), so a literal here carries no
# rounding risk -- unlike a derived constant. Deliberately NOT added to utils.py:
# that file is taken verbatim from sella-baseline and its sha256 is recorded in
# molecules/*.meta.json, so editing it would invalidate the baseline provenance.
KCAL_TO_KJ = 4.184

logger = logging.getLogger("validate_worker")
_SYSTEM_CACHE: dict[tuple[str, str, int], object] = {}
_OPTIMIZER_CACHE: dict[tuple[str, str | None, str, bool], object] = {}
DEFAULT_TASK_TIMEOUT_SECONDS = 2100.0
DEFAULT_MAX_TASK_RSS_GB = 32.0
DEFAULT_GUARD_POLL_SECONDS = 1.0
DEFAULT_KILL_GRACE_SECONDS = 5.0


def _backend_module_name(mode: str) -> str:
    if mode == "xtb":
        return "xtb_molecular_system"
    raise ValueError(f"Invalid mode: {mode}")


def _build_system(mode: str, baseline: dict, num_threads: int):
    module_name = _backend_module_name(mode)
    module = importlib.import_module(module_name)
    return module.MolecularSystem(
        baseline["xyz_path"], method="GFN2-xTB", num_threads=num_threads
    )


def _get_system(mode: str, baseline: dict, num_threads: int):
    cache_key = (mode, baseline["xyz_path"], int(num_threads))
    system = _SYSTEM_CACHE.get(cache_key)
    if system is None:
        system = _build_system(mode, baseline, num_threads)
        _SYSTEM_CACHE[cache_key] = system
    return system


def _cap_in_process_threads(num_threads: int) -> None:
    """Resize libgomp / openblas / MKL thread pools loaded in this process.

    OMP_NUM_THREADS only takes effect at library *load* time. Once
    libgomp is mapped (e.g. by numpy via openblas), env var
    changes are ignored — we have to call the runtime APIs via ctypes
    on the already-loaded handles. /proc/self/maps tells us which paths
    are loaded; RTLD_NOLOAD reuses the existing handles instead of
    re-dlopen'ing fresh copies.
    """
    num_threads = max(1, int(num_threads))
    _RTLD_NOLOAD = 0x00004

    loaded: dict[str, list[str]] = {"gomp": [], "openblas": [], "mkl": []}
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts or not parts[-1].startswith("/"):
                    continue
                p = parts[-1]
                if "libgomp" in p or "libomp" in p:
                    if p not in loaded["gomp"]:
                        loaded["gomp"].append(p)
                elif "libopenblas" in p:
                    if p not in loaded["openblas"]:
                        loaded["openblas"].append(p)
                elif "libmkl" in p:
                    if p not in loaded["mkl"]:
                        loaded["mkl"].append(p)
    except OSError:
        return  # /proc not available (e.g., macOS); env vars at load are all we get

    for path in loaded["gomp"]:
        try:
            lib = ctypes.CDLL(path, mode=_RTLD_NOLOAD)
            lib.omp_set_num_threads(num_threads)
        except (OSError, AttributeError):
            pass

    for path in loaded["openblas"]:
        try:
            lib = ctypes.CDLL(path, mode=_RTLD_NOLOAD)
            lib.openblas_set_num_threads(num_threads)
        except (OSError, AttributeError):
            pass

    for path in loaded["mkl"]:
        try:
            lib = ctypes.CDLL(path, mode=_RTLD_NOLOAD)
            lib.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_threads)))
        except (OSError, AttributeError):
            pass


def _preflight_backend_import(mode: str) -> None:
    module_name = _backend_module_name(mode)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import backend module {module_name!r} for mode {mode!r} "
            f"with interpreter {sys.executable}: {exc}"
        ) from exc
    if mode == "xtb":
        try:
            module._resolve_xtb_binary()
        except Exception as exc:
            raise RuntimeError(
                f"xtb binary not available for mode {mode!r}: {exc}"
            ) from exc


def _get_minimize_func(optimizer_spec: dict) -> object:
    cache_key = optimizer_cache_key(optimizer_spec)
    minimize_func = _OPTIMIZER_CACHE.get(cache_key)
    if minimize_func is None:
        minimize_func = load_minimize_func(optimizer_spec)
        _OPTIMIZER_CACHE[cache_key] = minimize_func
    return minimize_func


class _ForceCallLimit(BaseException):
    """Harness control flow, deliberately outside optimizer Exception handlers."""


class _OptimizerError(RuntimeError):
    error_kind = "optimizer_error"


class _InvalidOptimizerOutput(ValueError):
    error_kind = "optimizer_error"


class _InfrastructureError(RuntimeError):
    error_kind = "infrastructure_error"


def _error_payload(exc: BaseException) -> dict:
    return {
        "result": None,
        "error": str(exc) or type(exc).__name__,
        "error_kind": getattr(exc, "error_kind", "infrastructure_error"),
    }


def _write_task_checkpoint(task: dict, snapshot: dict) -> None:
    """Publish a complete snapshot atomically before returning from calc().

    The child may be killed during the next calculation or checkpoint write.
    A same-directory rename leaves the parent either complete prior snapshot or
    complete new snapshot; fsync keeps completed writes out of buffered Python IO.
    """
    path = task.get("_checkpoint_path")
    if path is None:
        return
    staging = path + ".tmp"
    try:
        with open(staging, "w") as stream:
            json.dump(snapshot, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    except OSError as exc:
        raise _InfrastructureError(f"checkpoint_io_error: {exc}") from exc


def _read_task_checkpoint(path: str) -> dict | None:
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def _run_optimization_task(task: dict) -> dict:
    capture = task.get("_diagnostic_capture")
    mode = task["mode"]
    _backend_module_name(mode)  # reject legacy force-field jobs before setup
    mol_name = task["mol_name"]
    baseline = task["baseline"]
    max_steps = operator.index(task["max_steps"])
    if isinstance(task["max_steps"], (bool, np.bool_)) or max_steps < 1:
        raise _InfrastructureError("max_steps must be a positive integer")
    num_threads = int(task.get("num_threads", 1))

    # Prepared references supply energies at the exact starting/final XYZ.
    # Do not evaluate a free initial or final force call outside the budget.
    initial_energy = float(baseline["initial_energy"])
    baseline_final_energy = float(baseline["final_energy"])
    baseline_improvement = float(baseline["improvement"])
    baseline_steps = operator.index(baseline["n_steps"])
    if (
        not all(math.isfinite(value) for value in (
            initial_energy, baseline_final_energy, baseline_improvement
        ))
        or baseline_improvement <= 0
        or baseline_steps <= 0
    ):
        raise _InfrastructureError("Invalid prepared reference energies or step count")

    system = _get_system(mode=mode, baseline=baseline, num_threads=num_threads)
    _cap_in_process_threads(num_threads)
    atomic_numbers = np.asarray(system.atomic_numbers).copy()
    conf_pos = np.asarray(system.initial_positions, dtype=float).copy()
    if (
        conf_pos.shape != (len(atomic_numbers), 3)
        or not np.all(np.isfinite(conf_pos))
    ):
        raise _InfrastructureError("Invalid prepared initial positions")
    if capture is not None:
        capture.call("initialize", atomic_numbers)
    try:
        minimize_func = _get_minimize_func(task["optimizer_spec"])
    except MemoryError:
        raise
    except BaseException as exc:
        raise _OptimizerError(f"optimizer_load_error: {exc}") from exc

    state = init_convergence_state(mode, conf_pos)
    state.record_metrics = capture is not None
    n_steps = 0
    n_completed_steps = 0
    last_calc_pos: np.ndarray | None = None
    last_calc_energy: float | None = None
    last_calc_converged = False
    calculation_error: BaseException | None = None

    def result_payload(stop_reason: str, *, converged: bool = False) -> dict:
        if last_calc_pos is None or last_calc_energy is None:
            raise _OptimizerError("returned_geometry_mismatch:no_successful_force_call")
        energy_improvement = initial_energy - last_calc_energy
        return {
            "result": {
                "mol_name": mol_name,
                "converged": converged,
                "stop_reason": stop_reason,
                "max_steps": max_steps,
                "n_steps": n_steps,
                "n_completed_steps": n_completed_steps,
                "final_energy": last_calc_energy,
                "final_positions": last_calc_pos.tolist(),
                "rel_energy": energy_improvement / baseline_improvement,
                "energy_delta_kcal_mol": (
                    last_calc_energy - baseline_final_energy
                ) / KCAL_TO_KJ,
                "rel_steps": n_steps / baseline_steps,
            },
            "error": None,
            "error_kind": None,
        }

    def checkpoint() -> None:
        payload = result_payload("time_limit") if last_calc_pos is not None else None
        _write_task_checkpoint(task, {"payload": payload})

    def invalidate(exc: BaseException) -> None:
        nonlocal calculation_error
        calculation_error = exc
        if capture is not None:
            capture.call("exception", exc)
        # A caught numerical/contract failure cannot be hidden by a later timeout.
        _write_task_checkpoint(task, {"payload": _error_payload(exc)})

    def calc(pos: np.ndarray, conv_check: bool = False):
        nonlocal n_steps, n_completed_steps
        nonlocal last_calc_pos, last_calc_energy, last_calc_converged
        if calculation_error is not None:
            raise calculation_error
        if conv_check:
            return is_converged(state)
        if n_steps >= max_steps:
            raise _ForceCallLimit()
        pos_arr = None
        try:
            pos_arr = np.asarray(pos, dtype=float).copy()
            if pos_arr.shape != conf_pos.shape:
                raise _InvalidOptimizerOutput("Wrong number of atoms or coordinates")
            if not np.all(np.isfinite(pos_arr)):
                raise _InvalidOptimizerOutput("Calculation positions are not finite")
        except (TypeError, ValueError) as exc:
            if capture is not None:
                capture.call("invalid_input", pos_arr)
            error = _InvalidOptimizerOutput(str(exc))
            invalidate(error)
            raise error from exc

        # Count every backend call started, even when a timeout interrupts it.
        # The saved geometry/energy remain tied to the last complete call.
        n_steps += 1
        if capture is not None:
            capture.call("attempted", pos_arr, n_steps, n_completed_steps)
        checkpoint()
        try:
            computed = system.compute(pos_arr.copy())
        except MemoryError as exc:
            error = _OptimizerError(f"numerical_memory_error: {exc}")
            invalidate(error)
            raise error from exc
        except ValueError as exc:
            # The xTB backend raises NumericalCalculationError (a ValueError)
            # for nonfinite results; malformed gradient parsing is also a
            # ValueError. These invalidate the calculation, not worker health.
            error = _InvalidOptimizerOutput(f"backend_numerical_error: {exc}")
            invalidate(error)
            raise error from exc
        except Exception as exc:
            error = _InfrastructureError(f"backend_calculation_error: {exc}")
            invalidate(error)
            raise error from exc
        try:
            raw_energy, raw_forces = computed
            energy_arr = np.asarray(raw_energy, dtype=float)
            forces = np.asarray(raw_forces, dtype=float).copy()
            if energy_arr.ndim != 0 or not np.isfinite(energy_arr):
                raise _InvalidOptimizerOutput("Calculation energy is not a finite scalar")
            if forces.shape != conf_pos.shape or not np.all(np.isfinite(forces)):
                raise _InvalidOptimizerOutput("Calculation forces are malformed or not finite")
            energy = float(energy_arr)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            error = _InvalidOptimizerOutput(str(exc))
            invalidate(error)
            raise error from exc
        last_calc_pos = pos_arr
        last_calc_energy = energy
        n_completed_steps += 1
        if capture is not None:
            capture.call("completed", pos_arr, forces, energy, n_steps, n_completed_steps)
        # Save the complete calculation before convergence bookkeeping, which
        # can itself be interrupted. Time-limited snapshots are nonconverged.
        checkpoint()
        try:
            update_convergence_state(state, pos_arr, energy, forces)
            last_calc_converged = bool(is_converged(state))
            if capture is not None and state.last_metrics is not None:
                capture.call("metrics", n_steps, state.last_metrics)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            error = _InvalidOptimizerOutput(str(exc))
            invalidate(error)
            raise error from exc
        return energy, forces

    def converged_from_calc(*_a, **_k):
        return is_converged(state)

    try:
        returned = minimize_func(
            conf_pos.copy(), atomic_numbers, calc,
            max_force_calls=max_steps, converged=converged_from_calc,
        )
    except _ForceCallLimit:
        if calculation_error is not None:
            raise calculation_error
        return result_payload("force_call_limit")
    except MemoryError as exc:
        raise _OptimizerError(f"numerical_memory_error: {exc}") from exc
    except (_OptimizerError, _InvalidOptimizerOutput, _InfrastructureError):
        raise
    except BaseException as exc:
        raise _OptimizerError(f"optimizer_exception: {exc}") from exc

    try:
        if calculation_error is not None:
            raise calculation_error
        try:
            opt_pos, n_steps_func = returned
            opt_pos_arr = np.asarray(opt_pos, dtype=float)
        except (TypeError, ValueError) as exc:
            raise _InvalidOptimizerOutput("Malformed optimizer return value") from exc
        if opt_pos_arr.shape != conf_pos.shape:
            raise _InvalidOptimizerOutput("Wrong number of atoms or coordinates")
        if not np.all(np.isfinite(opt_pos_arr)):
            raise _InvalidOptimizerOutput("Final positions are not finite")
        if last_calc_pos is None:
            raise _OptimizerError("returned_geometry_mismatch:no_successful_force_call")
        if not np.array_equal(opt_pos_arr, last_calc_pos):
            raise _OptimizerError(
                "returned_geometry_mismatch:last_evaluated_geometry_required"
            )
        if isinstance(n_steps_func, (bool, np.bool_)):
            raise _OptimizerError("force_call_count_mismatch:reported_count_is_not_an_integer")
        try:
            reported_n_steps = operator.index(n_steps_func)
        except TypeError as exc:
            raise _OptimizerError(
                "force_call_count_mismatch:reported_count_is_not_an_integer"
            ) from exc
        if reported_n_steps != n_steps:
            raise _OptimizerError(
                f"force_call_count_mismatch:measured={n_steps}:reported={reported_n_steps}"
            )
        stop_reason = (
            "converged" if last_calc_converged else
            "force_call_limit" if n_steps >= max_steps else "optimizer_returned"
        )
        return result_payload(stop_reason, converged=last_calc_converged)
    except MemoryError:
        raise
    except (_OptimizerError, _InvalidOptimizerOutput, _InfrastructureError):
        raise
    except BaseException as exc:
        raise _InvalidOptimizerOutput(f"Malformed optimizer return value: {exc}") from exc



def _read_rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _pids_in_process_group(pgid: int) -> list[int]:
    proc_dir = "/proc"
    try:
        names = os.listdir(proc_dir)
    except OSError:
        return [pgid]

    pids: list[int] = []
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(os.path.join(proc_dir, name, "stat")) as fh:
                stat = fh.read()
            rest = stat.rsplit(") ", 1)[1].split()
            proc_pgrp = int(rest[2])
        except (OSError, ValueError, IndexError):
            continue
        if proc_pgrp == pgid:
            pids.append(pid)
    return pids or [pgid]


def _process_group_rss_bytes(pgid: int) -> int:
    return sum(_read_rss_bytes(pid) for pid in _pids_in_process_group(pgid))


def _gib(value: int | float) -> float:
    return float(value) / (1024 ** 3)


def _kill_task_process_group(proc: mp.Process, *, grace_seconds: float) -> None:
    if proc.pid is None:
        return
    # Always deliver the second signal to the group: its leader may have exited
    # after SIGTERM while an xTB descendant still needs SIGKILL.
    for sig, wait_seconds in (
        (signal.SIGTERM, max(0.0, grace_seconds)),
        (signal.SIGKILL, 1.0),
    ):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            if not proc.is_alive():
                return
            proc.terminate() if sig == signal.SIGTERM else proc.kill()
        except OSError:
            if proc.is_alive():
                proc.terminate() if sig == signal.SIGTERM else proc.kill()
        proc.join(timeout=wait_seconds)


def _task_child_main(task: dict, result_queue: mp.Queue) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    # Only the parent publishes. Avoid advertising collector credentials/paths
    # to candidate code; OS-level worker containment is still required.
    for key in ("URL", "TOKEN_FILE", "SPOOL_DIR"):
        os.environ.pop("GIGAOPT_DIAGNOSTICS_" + key, None)
    capture = TaskCapture.create(task)
    task = {**task, "_diagnostic_capture": capture}
    try:
        result = _run_optimization_task(task)
    except BaseException as exc:
        if capture is not None:
            capture.call("exception", exc)
        result = _error_payload(exc)
        result["worker_guard"] = {
            "kind": "child_exception", "traceback": traceback.format_exc(),
        }
    finally:
        if capture is not None:
            capture.call("close")
    # Also persist final validation failures, so a queue/exit race does not turn
    # confirmed optimizer failure into a scoreable previous calculation.
    _write_task_checkpoint(task, {"payload": result, "finished": True})
    result_queue.put(result)


def _run_optimization_task_guarded(
    task: dict,
    *,
    task_timeout_seconds: float,
    max_task_rss_bytes: int,
    guard_poll_seconds: float = DEFAULT_GUARD_POLL_SECONDS,
    kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS,
    heartbeat=None,
) -> dict:
    """Run optimizer code only in a child, with durable last-calculation state."""
    start_time = time.monotonic()
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    result_queue = ctx.Queue(maxsize=1)
    checkpoint_dir = tempfile.TemporaryDirectory(prefix="optimization_checkpoint_")
    checkpoint_path = os.path.join(checkpoint_dir.name, "latest.json")
    child_task = {**task, "_checkpoint_path": checkpoint_path,
                  "_diagnostics_dir": os.path.join(checkpoint_dir.name, "diagnostics")}
    proc = ctx.Process(target=_task_child_main, args=(child_task, result_queue))
    max_seen_rss = 0
    kill_reason: str | None = None
    guard_poll_seconds = max(0.01, float(guard_poll_seconds))
    last_heartbeat = float("-inf")

    def finish(payload: dict) -> dict:
        # All numerical child work has finished or the guard has killed it.
        # Attach before TemporaryDirectory cleanup; never put arrays in Redis.
        return attach_bundle(child_task, payload)

    try:
        proc.start()
        while True:
            elapsed = time.monotonic() - start_time
            if heartbeat is not None and elapsed - last_heartbeat >= 5.0:
                heartbeat()
                last_heartbeat = elapsed
            # Drain while the child is alive; large coordinate payloads can block
            # its queue feeder and prevent exit if the parent waits for exit first.
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = None
            if result is not None:
                proc.join(timeout=1.0)
                return finish(result)
            if not proc.is_alive():
                break
            if max_task_rss_bytes > 0 and proc.pid is not None:
                rss = _process_group_rss_bytes(proc.pid)
                max_seen_rss = max(max_seen_rss, rss)
                if rss > max_task_rss_bytes:
                    kill_reason = (
                        "worker_guard_rss: task exceeded RSS limit "
                        f"rss={_gib(rss):.2f}GiB limit={_gib(max_task_rss_bytes):.2f}GiB "
                        f"elapsed={elapsed:.1f}s"
                    )
                    break
            if task_timeout_seconds > 0 and elapsed > task_timeout_seconds:
                kill_reason = (
                    "worker_guard_timeout: task exceeded wall-clock limit "
                    f"elapsed={elapsed:.1f}s limit={task_timeout_seconds:.1f}s "
                    f"max_rss={_gib(max_seen_rss):.2f}GiB"
                )
                break
            time.sleep(min(guard_poll_seconds, 5.0))

        if kill_reason is not None:
            _kill_task_process_group(proc, grace_seconds=kill_grace_seconds)
            proc.join(timeout=1.0)
            guard = {
                "kind": kill_reason.split(":", 1)[0],
                "elapsed_seconds": time.monotonic() - start_time,
                "max_rss_bytes": max_seen_rss,
                "max_task_rss_bytes": max_task_rss_bytes,
                "task_timeout_seconds": task_timeout_seconds,
            }
            snapshot = _read_task_checkpoint(checkpoint_path)
            payload = snapshot.get("payload") if snapshot else None
            if kill_reason.startswith("worker_guard_timeout:") and payload:
                if payload.get("error"):
                    return finish({**payload, "worker_guard": guard})
                if payload.get("result") is not None:
                    return finish({
                        **payload,
                        "result": {
                            **payload["result"], "converged": False,
                            "stop_reason": "time_limit",
                        },
                        "worker_guard": guard,
                    })
            return finish({
                **_error_payload(
                    _OptimizerError(kill_reason)
                    if guard["kind"] == "worker_guard_rss"
                    else _InfrastructureError(kill_reason)
                ),
                "worker_guard": guard,
            })

        proc.join(timeout=1.0)
        try:
            return finish(result_queue.get(timeout=0.1))
        except queue.Empty:
            snapshot = _read_task_checkpoint(checkpoint_path)
            if snapshot and snapshot.get("finished") and proc.exitcode == 0:
                return finish(snapshot["payload"])
            return finish({
                **_error_payload(_InfrastructureError(
                    "worker_child_exit: child exited without returning a result "
                    f"exitcode={proc.exitcode} max_rss={_gib(max_seen_rss):.2f}GiB"
                )),
                "worker_guard": {
                    "kind": "worker_child_exit", "exitcode": proc.exitcode,
                    "max_rss_bytes": max_seen_rss,
                },
            })
    finally:
        if proc.pid is not None:
            _kill_task_process_group(proc, grace_seconds=kill_grace_seconds)
        result_queue.close()
        result_queue.join_thread()
        checkpoint_dir.cleanup()


def _retry_diagnostic_upload(uploader) -> None:
    try:
        uploader.retry_pending()
    except Exception:
        logger.warning("Diagnostic upload retry unavailable; worker continues")


def _diagnostic_capture_enabled(uploader) -> bool:
    try:
        return bool(uploader.capture_enabled())
    except Exception:
        return False


def _publish_task_diagnostics(uploader, task: dict, payload: dict) -> dict:
    try:
        published = uploader.publish(task, payload)
        if not isinstance(published, dict):
            raise TypeError("invalid diagnostic publisher return")
        # Defense in depth: no private capture can reach Redis, even if a
        # collector/publisher misbehaves. This boundary never changes scores.
        published.pop("_diagnostic_bundle", None)
        return published
    except Exception:
        payload.pop("_diagnostic_bundle", None)
        payload["diagnostics"] = {
            "schema_version": 1, "availability": "unavailable",
            "reason": "diagnostic_publication_unavailable", "truncated": False,
        }
        return payload


def start_worker(
    redis_host: str,
    redis_port: int,
    supported_modes: list[str],
    num_threads: int = 1,
    poll_interval_seconds: float = 1.0,
    verbose: bool = False,
    log_dir: str = "logs",
    worker_name: str | None = None,
    max_tasks: int = 0,
    task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS,
    max_task_rss_gb: float = DEFAULT_MAX_TASK_RSS_GB,
    guard_poll_seconds: float = DEFAULT_GUARD_POLL_SECONDS,
    kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS,
) -> None:
    global logger
    from distributed_validate import diagnostic_upload
    if not supported_modes or any(mode != "xtb" for mode in supported_modes):
        raise ValueError("Workers support only xtb mode")
    worker_name = worker_name or f"validate_worker_{os.uname().nodename}_{os.getpid()}"
    logger = setup_logging(worker_name, verbose=verbose, log_dir=log_dir)

    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "redis package is required for distributed validation"
        ) from exc

    redis_conn = redis.Redis(
        host=redis_host, port=redis_port, socket_connect_timeout=5.0,
        socket_timeout=5.0,
    )
    try:
        redis_conn.ping()
    except redis.exceptions.RedisError as exc:
        raise RuntimeError(
            f"Redis is unavailable at {redis_host}:{redis_port}"
        ) from exc

    logger.info(
        "Starting validate worker %s for modes=%s",
        worker_name,
        ",".join(supported_modes),
    )
    logger.info("Python executable: %s", sys.executable)
    logger.info("Configured xTB threads per worker: %s", num_threads)
    max_task_rss_bytes = (
        int(max_task_rss_gb * (1024 ** 3)) if max_task_rss_gb > 0 else 0
    )
    logger.info(
        "Task guard: timeout=%ss max_rss=%sGiB poll=%ss kill_grace=%ss",
        task_timeout_seconds if task_timeout_seconds > 0 else "disabled",
        max_task_rss_gb if max_task_rss_bytes > 0 else "disabled",
        guard_poll_seconds,
        kill_grace_seconds,
    )

    # Override the conservative module-top defaults with the user-requested
    # thread count. Subprocesses (e.g. xtb) inherit these.
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    for mode in supported_modes:
        _preflight_backend_import(mode)
        logger.info("Backend import check passed for mode=%s", mode)

    # Resize thread pools that were already loaded by the imports above.
    # Without this, OpenBLAS keep their default (CPU count) pools
    # alive and burn cores busy-waiting between tasks.
    _cap_in_process_threads(num_threads)

    worker_id = f"{os.uname().nodename}:{os.getpid()}:{uuid.uuid4().hex}"
    lease_seconds = 30.0
    tasks_done = 0
    _retry_diagnostic_upload(diagnostic_upload)
    while True:
        task_id, task = load_next_optimization_task(
            redis_conn, supported_modes, worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        if task_id is None or task is None:
            _retry_diagnostic_upload(diagnostic_upload)
            time.sleep(poll_interval_seconds)
            continue

        logger.info(
            "Processing optimization task %s for %s %s using %s (xtb_threads=%s)",
            task_id,
            task["mode"],
            task["mol_name"],
            describe_optimizer_spec(task["optimizer_spec"]),
            num_threads,
        )

        task_error_logged = False
        try:
            task = {**task, "num_threads": num_threads, "task_id": task_id,
                    "_capture_diagnostics": _diagnostic_capture_enabled(diagnostic_upload)}
            result = _run_optimization_task_guarded(
                task,
                task_timeout_seconds=task_timeout_seconds,
                max_task_rss_bytes=max_task_rss_bytes,
                guard_poll_seconds=guard_poll_seconds,
                kill_grace_seconds=kill_grace_seconds,
                heartbeat=lambda: heartbeat_worker(
                    redis_conn, worker_id, task_id, lease_seconds=lease_seconds
                ),
            )
        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc)
            logger.debug(traceback.format_exc())
            task_error_logged = True
            result = _error_payload(exc)
        if result.get("error") and not task_error_logged:
            logger.error("Task %s failed: %s", task_id, result["error"])

        result = _publish_task_diagnostics(diagnostic_upload, task, result)
        store_optimization_result(redis_conn, task_id, result)
        finish_worker_task(redis_conn, worker_id, task_id)
        _retry_diagnostic_upload(diagnostic_upload)

        # Bound native-memory growth by exiting cleanly after a fixed
        # number of tasks; the launcher respawns us.
        tasks_done += 1
        if max_tasks and tasks_done >= max_tasks:
            logger.info(
                "Processed %d tasks (limit %d); exiting for respawn.",
                tasks_done,
                max_tasks,
            )
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed validation worker")
    parser.add_argument(
        "--redis-host", type=str, default="localhost", help="Redis hostname or IP"
    )
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["xtb"],
        default="xtb",
        help="Worker backend to serve",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="xTB threads to use per worker process",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Sleep duration when no tasks are available",
    )
    parser.add_argument(
        "--verbose",
        type=str_to_bool,
        default=False,
        help="Enable verbose (debug) logging (true/false)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to store log files"
    )
    parser.add_argument("--worker-name", type=str, default=None, help="Worker name")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help=(
            "Exit cleanly after processing this many tasks "
            "(0 = unbounded). Used with a launcher-side restart loop to "
            "bound per-worker memory."
        ),
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help=(
            "Kill a single optimization task after this many seconds "
            f"(default: {DEFAULT_TASK_TIMEOUT_SECONDS}; 0 = disabled)"
        ),
    )
    parser.add_argument(
        "--max-rss-gb",
        type=float,
        default=DEFAULT_MAX_TASK_RSS_GB,
        help=(
            "Kill a single optimization task when its process group exceeds "
            f"this RSS in GiB (default: {DEFAULT_MAX_TASK_RSS_GB}; 0 = disabled)"
        ),
    )
    parser.add_argument(
        "--guard-poll-interval",
        type=float,
        default=DEFAULT_GUARD_POLL_SECONDS,
        help=(
            "Seconds between task guard checks "
            f"(default: {DEFAULT_GUARD_POLL_SECONDS})"
        ),
    )
    parser.add_argument(
        "--kill-grace",
        type=float,
        default=DEFAULT_KILL_GRACE_SECONDS,
        help=(
            "Seconds to wait after SIGTERM before SIGKILL "
            f"(default: {DEFAULT_KILL_GRACE_SECONDS})"
        ),
    )

    args = parser.parse_args()
    supported_modes = [args.mode]
    start_worker(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        supported_modes=supported_modes,
        num_threads=args.num_threads,
        poll_interval_seconds=args.poll_interval,
        verbose=args.verbose,
        log_dir=args.log_dir,
        worker_name=args.worker_name,
        max_tasks=args.max_tasks,
        task_timeout_seconds=args.task_timeout,
        max_task_rss_gb=args.max_rss_gb,
        guard_poll_seconds=args.guard_poll_interval,
        kill_grace_seconds=args.kill_grace,
    )
