"""Check exact source export and preservation of immutable release identities."""
from pathlib import Path
import subprocess

import pytest

from scripts import export_dataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("source_commit,profile", export_dataset.SOURCE_PROFILES.items())
def test_export_reproduces_every_frozen_release_byte(source_commit, profile):
    available = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", source_commit],
        capture_output=True,
    )
    if available.returncode:
        pytest.skip("Source commit objects are unavailable in this checkout")
    payload = export_dataset.build_release(ROOT, source_commit)
    frozen = ROOT / "molecules" / "releases" / profile["release_id"]
    if not frozen.is_dir():
        pytest.skip("Historical release not shipped in this branch")
    assert set(payload) == {path.relative_to(frozen).as_posix() for path in frozen.rglob("*") if path.is_file()}
    for name, expected in payload.items():
        assert (frozen / name).read_bytes() == expected, name


def test_existing_release_is_never_overwritten(tmp_path, monkeypatch):
    destination = tmp_path / export_dataset.RELEASE_ID
    destination.mkdir()
    (destination / "marker").write_bytes(b"original")
    monkeypatch.setattr(export_dataset, "build_release", lambda *args: pytest.fail("must refuse before exporting"))
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        export_dataset.export(ROOT, destination)
    assert (destination / "marker").read_bytes() == b"original"


def test_export_failure_cleans_only_its_new_partial_release(tmp_path, monkeypatch):
    destination = tmp_path / export_dataset.RELEASE_ID
    unrelated = tmp_path / "other-release"
    unrelated.mkdir()
    (unrelated / "marker").write_bytes(b"preserve")
    monkeypatch.setattr(export_dataset, "build_release", lambda *args: {
        "first": b"1", "second": b"2",
        "release.json": export_dataset.json_bytes({"dataset_release_id": export_dataset.RELEASE_ID}),
    })
    original_rename = export_dataset.os.rename
    calls = 0

    def fail_second_rename(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interrupted export")
        original_rename(source, target)

    monkeypatch.setattr(export_dataset.os, "rename", fail_second_rename)
    with pytest.raises(OSError, match="interrupted export"):
        export_dataset.export(ROOT, destination)
    assert not destination.exists()
    assert (unrelated / "marker").read_bytes() == b"preserve"
    assert not list(tmp_path.glob(".dataset-export-*"))
