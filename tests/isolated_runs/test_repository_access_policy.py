"""Provider argv checks only: no model calls, Docker, or network requests."""
from __future__ import annotations

import tomllib

import pytest

from isolated_runs import runner


def configuration(provider):
    return {"provider": provider, "model": "gpt-6-astra" if provider == "codex" else "claude-opus-5",
            "effort": "high" if provider == "codex" else "max", "run_id": "own-run"}


def command_for(provider, purpose):
    cfg = configuration(provider)
    if purpose == "research":
        return runner.ralph_command(cfg)
    return runner.verification_command(cfg, "A bounded verification prompt.")


def policy_from_command(command, provider):
    if provider == "codex":
        values = [command[i + 1] for i, value in enumerate(command) if value == "--config"]
        overrides = [tomllib.loads(value) for value in values]
        policies = [value["developer_instructions"] for value in overrides if "developer_instructions" in value]
        assert len(policies) == 1
        return policies[0]
    assert command.count("--append-system-prompt") == 1
    return command[command.index("--append-system-prompt") + 1]


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("purpose", ["research", "verification"])
def test_all_provider_paths_receive_repository_restriction_and_preserve_flags(provider, purpose):
    command = command_for(provider, purpose)
    policy = policy_from_command(command, provider)
    assert policy == runner.REPOSITORY_ACCESS_POLICY.read_text().strip()
    assert "https://" not in policy
    for phrase in ("Do not access, clone, download, search or reuse", "every branch, tag, commit",
                   "external copies of this", "public and anonymous mirrors",
                   "raw files, APIs, archives, mirrors, forks, caches and indexed excerpts",
                   "Never bypass", "Ignore unexpectedly exposed content",
                   "approved workspace/data remain permitted", "unrelated public",
                   "repositories, papers, documentation and research sources"):
        assert phrase in policy
    cfg = configuration(provider)
    assert command[command.index("--model") + 1] == cfg["model"]
    if provider == "codex":
        assert 'model_reasoning_effort="high"' in command
        assert "--dangerously-bypass-approvals-and-sandbox" in command
        assert command[command.index("--cd") + 1] == "/workspace"
    else:
        assert command[command.index("--effort") + 1] == "max"
        assert "--dangerously-skip-permissions" in command
        assert "--system-prompt" not in command
    if purpose == "research":
        assert command[0] == "ralph" and command[command.index("--max-iterations") + 1] == "0"
        assert "--no-commit" in command and "--no-questions" in command and "--no-allow-all" in command
        policy_index = next(i for i, value in enumerate(command)
                            if value.startswith("developer_instructions=") or value == "--append-system-prompt")
        assert policy_index > command.index("--")  # These arguments belong to the provider, not Ralph.
    else:
        assert command.count("A bounded verification prompt.") == 1
        assert command[0] == provider and "ralph" not in command and "--resume" not in command
        assert "--json" in command if provider == "codex" else "stream-json" in command


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("purpose", ["research", "verification"])
@pytest.mark.parametrize("contents", [None, " \n\t", b"\xff"])
def test_missing_empty_or_invalid_runtime_policy_blocks_provider_commands(tmp_path, monkeypatch, provider, purpose, contents):
    path = tmp_path / "repository_access_policy.md"
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    elif contents is not None:
        path.write_text(contents)
    monkeypatch.setattr(runner, "REPOSITORY_ACCESS_POLICY", path)
    with pytest.raises(runner.GateError, match="repository access policy.*provider was not started"):
        command_for(provider, purpose)


def test_codex_policy_toml_quoting_preserves_quotes_newlines_and_unicode(tmp_path, monkeypatch):
    text = 'Instruction "with quotes" and \\paths\nUnicode: \u03b1 \U0001f9ea\nA literal shell example: $(false) `false`\tDone.'
    path = tmp_path / "repository_access_policy.md"
    path.write_text(text)
    monkeypatch.setattr(runner, "REPOSITORY_ACCESS_POLICY", path)
    assert policy_from_command(runner.repository_policy_arguments("codex"), "codex") == text


def test_policy_is_baked_next_to_runner_and_ignores_workspace_file(tmp_path, monkeypatch):
    expected = runner.REPOSITORY_ACCESS_POLICY.read_text().strip()
    (tmp_path / "repository_access_policy.md").write_text("A run cannot replace the runtime policy.")
    monkeypatch.chdir(tmp_path)
    assert runner.REPOSITORY_ACCESS_POLICY.parent == runner.Path(runner.__file__).resolve().parent
    assert policy_from_command(runner.repository_policy_arguments("claude"), "claude") == expected
