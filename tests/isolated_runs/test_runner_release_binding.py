"""Run receipts must use the dataset release approved for the mounted run."""
import pytest

from isolated_runs.runner import GateError, check_anchor_identity, environment


def config():
    return {"run_id": "run-a", "initial_commit": "c" * 40,
            "initial_source_sha256": "a" * 64, "expected_release_id": "approved-release",
            "evaluation": {"user": "worker", "host": "fixture", "port": 22},
            "provider": "codex"}


def receipt():
    return {"status": "complete", "split": "train", "run_id": "run-a", "candidate_commit": "c" * 40,
            "source_sha256": "a" * 64, "program_id": "a" * 64, "evaluation_id": "e" * 32,
            "release_id": "approved-release", "dataset_release_id": "approved-release",
            "timestamp": "2026-09-10T00:00:00Z", "is_valid": 1}


@pytest.mark.parametrize("change", [
    {"release_id": "older-release", "dataset_release_id": "older-release"},
    {"dataset_release_id": "older-release"},
])
def test_valid_scientific_anchor_from_stale_gateway_cannot_start_research(change):
    check_anchor_identity(receipt(), "train", config())
    with pytest.raises(GateError, match="dataset release"):
        check_anchor_identity({**receipt(), **change}, "train", config())


def test_legacy_runtime_without_approved_release_cannot_start():
    legacy = config()
    del legacy["expected_release_id"]
    with pytest.raises(GateError, match="approved dataset release"):
        environment(legacy)
    with pytest.raises(GateError, match="approved dataset release"):
        check_anchor_identity(receipt(), "train", legacy)


def test_child_evaluations_inherit_the_approved_release(monkeypatch):
    monkeypatch.setenv("EVALUATION_EXPECTED_RELEASE_ID", "stale-parent-value")
    assert environment(config())["EVALUATION_EXPECTED_RELEASE_ID"] == "approved-release"


def test_child_git_trust_is_limited_to_the_mounted_workspace(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "*")
    env = environment(config())
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert env["GIT_CONFIG_VALUE_0"] == "/workspace"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
