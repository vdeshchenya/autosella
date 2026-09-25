"""Tests for distributed_validate.optimizer — spec normalization,
serialization round-trips, cache keys, describe string."""

from __future__ import annotations

import pickle

import pytest

from distributed_validate.optimizer import (
    describe_optimizer_spec,
    load_minimize_func,
    normalize_optimizer_spec,
    optimizer_cache_key,
    serialize_minimize_func,
)


# ── normalize_optimizer_spec ─────────────────────────────────────────────────

def test_normalize_from_string():
    spec = normalize_optimizer_spec("sample_optimizer")
    assert spec["kind"] == "importable"
    assert spec["module_name"] == "sample_optimizer"
    assert spec["use_entrypoint"] is True
    assert spec["entrypoint_name"] == "entrypoint"


def test_normalize_from_bytes():
    raw = pickle.dumps({"marker": "fake_payload"})
    spec = normalize_optimizer_spec(raw)
    assert spec["kind"] == "pickled"
    assert spec["payload"] == raw
    assert len(spec["payload_sha256"]) == 64  # hex sha256
    assert spec["source"] == "bytes"


def test_normalize_from_memoryview():
    raw = pickle.dumps({"marker": "fake_payload"})
    spec = normalize_optimizer_spec(memoryview(raw))
    assert spec["kind"] == "pickled"
    assert spec["payload"] == raw


def test_normalize_from_empty_bytes_raises():
    with pytest.raises(ValueError, match="empty"):
        normalize_optimizer_spec(b"")


def test_normalize_from_dict_importable():
    spec = normalize_optimizer_spec({"module_name": "sample_optimizer"})
    assert spec["kind"] == "importable"
    assert spec["use_entrypoint"] is True


def test_normalize_from_dict_with_function_name():
    spec = normalize_optimizer_spec(
        {"module_name": "sample_optimizer", "function_name": "minimize_func"}
    )
    assert spec["kind"] == "importable"
    assert spec["function_name"] == "minimize_func"
    assert spec["use_entrypoint"] is False


def test_normalize_from_dict_pickled_payload():
    raw = pickle.dumps({"marker": "fake_payload"})
    spec = normalize_optimizer_spec({"kind": "pickled", "payload": raw})
    assert spec["kind"] == "pickled"
    assert spec["payload"] == raw


def test_normalize_from_dict_missing_module_name_raises():
    with pytest.raises(ValueError, match="module_name"):
        normalize_optimizer_spec({"kind": "importable"})


def test_normalize_from_dict_no_entrypoint_no_function_raises():
    with pytest.raises(ValueError, match="function_name"):
        normalize_optimizer_spec({
            "module_name": "some.module",
            "use_entrypoint": False,
        })


def test_normalize_from_invalid_pickled_dict_raises():
    with pytest.raises(ValueError, match="bytes-like"):
        normalize_optimizer_spec({"kind": "pickled", "payload": "not bytes"})


def test_normalize_from_callable_importable():
    """A function defined in a real module should yield an importable spec."""
    from sample_optimizer import minimize_func
    spec = normalize_optimizer_spec(minimize_func)
    assert spec["kind"] == "importable"
    assert spec["module_name"] == "sample_optimizer"
    assert spec["function_name"] == "minimize_func"
    assert spec["use_entrypoint"] is False


def test_normalize_from_lambda_falls_back_to_pickled():
    """Lambdas have __qualname__ containing '<lambda>' / '<locals>' → pickled."""
    spec = normalize_optimizer_spec(lambda: 1)
    assert spec["kind"] == "pickled"
    assert spec["source"] == "callable"


def test_normalize_from_nested_function_falls_back_to_pickled():
    def outer():
        def inner():
            return 1
        return inner

    spec = normalize_optimizer_spec(outer())
    # qualname contains "<locals>" → pickled fallback
    assert spec["kind"] == "pickled"


def test_normalize_unsupported_type_raises():
    with pytest.raises(TypeError):
        normalize_optimizer_spec(12345)


# ── optimizer_cache_key ──────────────────────────────────────────────────────

def test_cache_key_deterministic_for_importable():
    k1 = optimizer_cache_key({"module_name": "sample_optimizer"})
    k2 = optimizer_cache_key({"module_name": "sample_optimizer"})
    assert k1 == k2


def test_cache_key_differs_between_modules():
    k1 = optimizer_cache_key({"module_name": "foo.bar"})
    k2 = optimizer_cache_key({"module_name": "baz.qux"})
    assert k1 != k2


def test_cache_key_for_pickled_uses_sha256():
    raw = pickle.dumps({"marker": "fake_payload"})
    spec = normalize_optimizer_spec(raw)
    key = optimizer_cache_key(spec)
    assert key[0].startswith("pickled:")
    # Same bytes → same key
    assert optimizer_cache_key(normalize_optimizer_spec(raw)) == key


# ── describe_optimizer_spec ──────────────────────────────────────────────────

def test_describe_importable_with_entrypoint():
    s = describe_optimizer_spec({"module_name": "sample_optimizer"})
    assert "sample_optimizer" in s
    assert "entrypoint" in s


def test_describe_importable_explicit_function():
    s = describe_optimizer_spec({
        "module_name": "sample_optimizer",
        "function_name": "minimize_func",
    })
    assert "minimize_func" in s


def test_describe_pickled_shows_hash_prefix():
    raw = pickle.dumps({"marker": "fake_payload"})
    s = describe_optimizer_spec(normalize_optimizer_spec(raw))
    assert s.startswith("pickled:")
    assert len(s) > len("pickled:")


# ── load_minimize_func round-trips ──────────────────────────────────────────

def test_load_from_importable_with_entrypoint():
    """`sample_optimizer` defines `entrypoint()` — load via entrypoint."""
    fn = load_minimize_func({"module_name": "sample_optimizer"})
    assert callable(fn)
    # Should return minimize_func
    from sample_optimizer import minimize_func
    assert fn is minimize_func


def test_load_from_importable_with_function_name():
    fn = load_minimize_func({
        "module_name": "sample_optimizer",
        "function_name": "minimize_func",
    })
    from sample_optimizer import minimize_func
    assert fn is minimize_func


def test_load_from_pickled_round_trip():
    """Serialize then load → get an equivalent callable back."""
    from sample_optimizer import minimize_func
    raw = serialize_minimize_func(minimize_func)
    fn = load_minimize_func(normalize_optimizer_spec(raw))
    assert callable(fn)


def test_load_missing_module_raises():
    with pytest.raises(Exception):  # ModuleNotFoundError, wrapped or not
        load_minimize_func({"module_name": "nonexistent.module.foo"})


def test_load_module_without_entrypoint_raises():
    """utils.py has no `entrypoint()` function."""
    with pytest.raises(RuntimeError, match="entrypoint"):
        load_minimize_func({"module_name": "utils"})


def test_load_module_without_named_function_raises():
    with pytest.raises(RuntimeError):
        load_minimize_func({
            "module_name": "utils",
            "function_name": "does_not_exist",
            "use_entrypoint": False,
        })


def test_serialize_returns_bytes():
    from sample_optimizer import minimize_func
    raw = serialize_minimize_func(minimize_func)
    assert isinstance(raw, bytes)
    assert len(raw) > 0
