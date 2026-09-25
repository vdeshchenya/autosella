import hashlib
import json
from pathlib import Path

import pytest

from scripts.install_dataset import install, verify


def release(path):
    path.mkdir()
    (path / "geometry.xyz").write_text("prepared bytes")
    hashes = {"geometry.xyz": hashlib.sha256(b"prepared bytes").hexdigest()}
    (path / "sha256.json").write_text(json.dumps(hashes))
    (path / "release.json").write_text(json.dumps({
        "dataset_release_id": "test-release", "checksums_file": "sha256.json",
        "checksums_sha256": hashlib.sha256((path / "sha256.json").read_bytes()).hexdigest()}))
    return path


def test_install_is_idempotent_and_read_only(tmp_path):
    source = release(tmp_path / "source")
    destination = tmp_path / "installed" / "test-release"
    assert install(source, destination) == destination
    assert install(source, destination) == destination
    assert not (destination / "geometry.xyz").stat().st_mode & 0o222
    verify(destination)


def test_corrupted_source_cannot_be_installed(tmp_path):
    source = release(tmp_path / "source")
    (source / "geometry.xyz").write_text("different bytes")
    destination = tmp_path / "test-release"
    with pytest.raises(ValueError, match="payload mismatch"):
        install(source, destination)
    assert not destination.exists()


def test_existing_different_release_is_never_overwritten(tmp_path):
    source = release(tmp_path / "source")
    destination = release(tmp_path / "test-release")
    manifest = json.loads((destination / "release.json").read_text())
    manifest["setting"] = "other"
    (destination / "release.json").write_text(json.dumps(manifest))
    before = (destination / "release.json").read_bytes()
    with pytest.raises(ValueError, match="refusing overwrite"):
        install(source, destination)
    assert (destination / "release.json").read_bytes() == before
