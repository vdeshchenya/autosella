from __future__ import annotations

import ast
import hashlib
import importlib
import linecache
import pickle
import sys
import types
from typing import Any, Callable


def _import_cloudpickle():
    try:
        import cloudpickle
    except ImportError:
        return None
    return cloudpickle


def serialize_minimize_func(minimize_func: Callable) -> bytes:
    cloudpickle = _import_cloudpickle()
    if cloudpickle is not None:
        return cloudpickle.dumps(minimize_func)

    try:
        return pickle.dumps(minimize_func)
    except Exception as exc:
        raise RuntimeError(
            "Could not pickle minimize_func. Install cloudpickle to evaluate "
            "non-importable or notebook-defined functions in distributed mode."
        ) from exc


def _normalize_pickled_optimizer_spec(
    payload: bytes,
    *,
    source: str = "bytes",
) -> dict[str, Any]:
    payload_bytes = bytes(payload)
    if not payload_bytes:
        raise ValueError("pickled optimizer payload is empty")

    return {
        "kind": "pickled",
        "payload": payload_bytes,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "source": source,
    }


def normalize_optimizer_spec(optimizer: Any) -> dict[str, Any]:
    if isinstance(optimizer, (bytes, bytearray, memoryview)):
        return _normalize_pickled_optimizer_spec(bytes(optimizer))

    if isinstance(optimizer, str):
        return {
            "kind": "importable",
            "module_name": optimizer,
            "entrypoint_name": "entrypoint",
            "use_entrypoint": True,
            "function_name": None,
        }

    if isinstance(optimizer, dict):
        if optimizer.get("kind") == "source":
            source = optimizer.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source optimizer must include non-empty UTF-8 source")
            if len(source.encode("utf-8")) > 1024 * 1024:
                raise ValueError("source optimizer exceeds 1 MiB")
            # Static syntax validation only. Candidate code must never be imported
            # or deserialized on the trusted evaluation coordinator.
            ast.parse(source, filename="<optimizer-source>")
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if optimizer.get("source_sha256", digest) != digest:
                raise ValueError("source optimizer hash mismatch")
            function_name = optimizer.get("function_name", "minimize_func")
            if function_name != "minimize_func":
                raise ValueError("source optimizer function must be minimize_func")
            return {"kind": "source", "source": source,
                    "source_sha256": digest, "function_name": function_name}
        if optimizer.get("kind") == "pickled" or "payload" in optimizer:
            payload = optimizer.get("payload")
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise ValueError(
                    "pickled optimizer spec must include bytes-like 'payload'"
                )
            source = str(optimizer.get("source", "dict"))
            return _normalize_pickled_optimizer_spec(bytes(payload), source=source)

        module_name = optimizer.get("module_name")
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("optimizer spec must include a non-empty 'module_name'")

        function_name = optimizer.get("function_name")
        entrypoint_name = optimizer.get("entrypoint_name", "entrypoint")
        use_entrypoint = bool(
            optimizer.get("use_entrypoint", function_name is None)
        )

        if not use_entrypoint and not function_name:
            raise ValueError(
                "optimizer spec must include 'function_name' when 'use_entrypoint' is false"
            )

        return {
            "kind": "importable",
            "module_name": module_name,
            "function_name": function_name,
            "entrypoint_name": entrypoint_name,
            "use_entrypoint": use_entrypoint,
        }

    if callable(optimizer):
        module_name = getattr(optimizer, "__module__", None)
        qualname = getattr(optimizer, "__qualname__", "")
        function_name = getattr(optimizer, "__name__", None)

        if not module_name or module_name == "__main__":
            return _normalize_pickled_optimizer_spec(
                serialize_minimize_func(optimizer),
                source="callable",
            )

        if "<locals>" in qualname:
            return _normalize_pickled_optimizer_spec(
                serialize_minimize_func(optimizer),
                source="callable",
            )

        if not function_name:
            raise RuntimeError("Could not determine optimizer function name")

        return {
            "kind": "importable",
            "module_name": module_name,
            "function_name": function_name,
            "entrypoint_name": "entrypoint",
            "use_entrypoint": False,
        }

    raise TypeError(
        "optimizer must be a module path string, optimizer spec dict, or callable"
    )


def optimizer_cache_key(spec: dict[str, Any]) -> tuple[str, str | None, str, bool]:
    normalized = normalize_optimizer_spec(spec)
    if normalized["kind"] == "source":
        return (f"source:{normalized['source_sha256']}", "minimize_func", "", False)
    if normalized["kind"] == "pickled":
        return (
            f"pickled:{normalized['payload_sha256']}",
            None,
            "",
            False,
        )
    return (
        normalized["module_name"],
        normalized["function_name"],
        normalized["entrypoint_name"],
        bool(normalized["use_entrypoint"]),
    )


def describe_optimizer_spec(spec: dict[str, Any]) -> str:
    normalized = normalize_optimizer_spec(spec)
    if normalized["kind"] == "source":
        return f"source:{normalized['source_sha256'][:12]}:minimize_func"
    if normalized["kind"] == "pickled":
        return f"pickled:{normalized['payload_sha256'][:12]}"
    if normalized["use_entrypoint"]:
        return (
            f"{normalized['module_name']}:{normalized['entrypoint_name']}()"
        )
    return f"{normalized['module_name']}:{normalized['function_name']}"


def load_minimize_func(spec: dict[str, Any]) -> Callable:
    """Load an optimizer; source specs may only be loaded inside a worker child."""
    normalized = normalize_optimizer_spec(spec)
    if normalized["kind"] == "source":
        module_name = f"_optimizer_source_{normalized['source_sha256']}"
        module = types.ModuleType(module_name)
        module.__file__ = f"<{module_name}>"
        sys.modules[module_name] = module
        # Tracebacks from source-loaded candidates must resolve against the exact
        # hashed source rather than an unrelated on-disk algo.py.
        linecache.cache[module.__file__] = (
            len(normalized["source"]), None,
            normalized["source"].splitlines(keepends=True), module.__file__,
        )
        try:
            exec(compile(normalized["source"], module.__file__, "exec"), module.__dict__)
            minimize_func = getattr(module, "minimize_func", None)
            if not callable(minimize_func):
                raise RuntimeError("Source optimizer must define callable minimize_func")
            return minimize_func
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    if normalized["kind"] == "pickled":
        cloudpickle = _import_cloudpickle()
        try:
            if cloudpickle is not None:
                minimize_func = cloudpickle.loads(normalized["payload"])
            else:
                minimize_func = pickle.loads(normalized["payload"])
        except Exception as exc:
            raise RuntimeError(
                "Could not unpickle distributed minimize_func payload. "
                "If it was created with cloudpickle, ensure cloudpickle is installed "
                "in the worker environment."
            ) from exc

        if minimize_func is None or not callable(minimize_func):
            raise RuntimeError(
                f"Pickled optimizer {describe_optimizer_spec(normalized)} "
                "did not deserialize to a callable"
            )

        return minimize_func

    module = importlib.import_module(normalized["module_name"])

    if normalized["use_entrypoint"]:
        entrypoint = getattr(module, normalized["entrypoint_name"], None)
        if entrypoint is None or not callable(entrypoint):
            raise RuntimeError(
                f"Module {normalized['module_name']!r} does not define callable "
                f"{normalized['entrypoint_name']!r}"
            )
        minimize_func = entrypoint()
    else:
        minimize_func = getattr(module, normalized["function_name"], None)

    if minimize_func is None or not callable(minimize_func):
        raise RuntimeError(
            f"Could not load callable optimizer from {describe_optimizer_spec(normalized)}"
        )

    return minimize_func
