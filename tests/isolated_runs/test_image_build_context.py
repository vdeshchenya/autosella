"""Exercise the installer context without Docker, network, or provider sessions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_installer_minimal_context_can_construct_every_provider_command(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    # Inspect the actual disposable context while the installer owns it. Import
    # its runner in isolation, so a policy in the checkout cannot mask omission.
    docker.write_text(f"#!{sys.executable}\n" + '''
import json
import os
from pathlib import Path
import sys

if sys.argv[1] == "build":
    context = Path(sys.argv[-1]).resolve()
    sys.path = [str(context)] + [p for p in sys.path if p and "opt_problem" not in p]
    from isolated_runs import runner
    assert Path(runner.__file__).is_relative_to(context)
    policy = runner.REPOSITORY_ACCESS_POLICY.read_text().strip()
    commands = []
    for provider in ("codex", "claude"):
        config = {"provider": provider, "model": "explicit-model", "effort": "high", "run_id": "own-run"}
        for command in (runner.ralph_command(config), runner.verification_command(config, "Verify only.")):
            assert any(policy in value or "developer_instructions=" in value for value in command)
            commands.append(command)
    Path(os.environ["AR_TEST_BUILD_RECEIPT"]).write_text(json.dumps({
        "files": sorted(str(p.relative_to(context)) for p in context.rglob("*") if p.is_file()),
        "policy": policy, "commands": commands,
    }))
elif sys.argv[1:3] == ["image", "inspect"]:
    print("sha256:" + "a" * 64)
else:
    raise AssertionError(sys.argv)
''')
    docker.chmod(0o755)
    receipt = tmp_path / "receipt.json"
    env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "", "AR_TEST_BUILD_RECEIPT": str(receipt),
           "BASE_IMAGE": "node:reviewed@sha256:" + "a" * 64, "BUN_VERSION": "1.2.3",
           "RALPH_COMMIT": "b" * 40, "CODEX_VERSION": "1.2.3", "CLAUDE_VERSION": "1.2.3",
           "IMAGE_TAG": "ar-test:context-only"}
    subprocess.run(["bash", str(repo / "scripts/install_autoresearch_tools.sh")],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    built = json.loads(receipt.read_text())
    assert built["files"] == [
        "Dockerfile", "evaluation_access/__init__.py", "evaluation_access/client.py",
        "isolated_runs/__init__.py", "isolated_runs/network.py", "isolated_runs/patch_ralph.py",
        "isolated_runs/repository_access_policy.md", "isolated_runs/runner.py",
    ]
    assert built["policy"] == (repo / "isolated_runs/repository_access_policy.md").read_text().strip()
    assert len(built["commands"]) == 4
