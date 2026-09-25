from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from isolated_runs import runner


def test_ralph_logs_rotate_across_restarts_and_preserve_unrelated_files(tmp_path):
    first = runner.RalphLog(tmp_path, max_bytes=5, keep=2)
    first.write(b"1234567890abcd")
    first.close()
    directory = tmp_path / ".ralph/logs"
    assert [p.read_bytes() for p in sorted(directory.glob("ralph-*.log"))] == [b"67890", b"abcd"]
    (directory / "notes.txt").write_text("keep")
    second = runner.RalphLog(tmp_path, max_bytes=5, keep=2)
    second.write(b"new")
    second.close()
    assert [p.read_bytes() for p in sorted(directory.glob("ralph-*.log"))] == [b"abcd", b"new"]
    assert (directory / "notes.txt").read_text() == "keep"


@pytest.mark.parametrize("component", [".ralph", ".ralph/logs"])
def test_ralph_log_rejects_symlink_components(tmp_path, component):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / component
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        runner.RalphLog(tmp_path)
    assert not list(outside.iterdir())


def test_ralph_tee_persists_stdout_stderr_and_returns_exit(tmp_path, monkeypatch, capfd):
    monkeypatch.setattr(runner, "ralph_command", lambda _: [sys.executable, "-c",
        "import sys; print('stdout evidence'); print('stderr evidence',file=sys.stderr); sys.exit(7)"])
    assert runner.run_ralph({}, dict(os.environ), workspace=tmp_path) == 7
    saved = b"".join(p.read_bytes() for p in sorted((tmp_path / ".ralph/logs").glob("*.log")))
    assert b"stdout evidence" in saved and b"stderr evidence" in saved
    assert "stdout evidence" in capfd.readouterr().out


def test_ralph_stop_signal_reaches_child_and_keeps_last_output(tmp_path):
    script = """import os,sys
from pathlib import Path
from isolated_runs import runner
runner.ralph_command=lambda _: [sys.executable,'-u','-c',"import time;print('started',flush=True);time.sleep(60)"]
raise SystemExit(runner.run_ralph({},dict(os.environ),workspace=Path(sys.argv[1])))
"""
    process = subprocess.Popen([sys.executable, "-u", "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=Path(__file__).resolve().parents[2])
    try:
        assert process.stdout.readline() == b"started\n"
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=8)
        assert process.returncode == 143
        assert b"started" in next((tmp_path / ".ralph/logs").glob("*.log")).read_bytes()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
