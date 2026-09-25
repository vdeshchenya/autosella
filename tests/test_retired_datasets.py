"""Withdrawn datasets must fail before launch or gateway side effects."""
import hashlib
import json

import pytest

from evaluation_access.release_policy import CURRENT_RELEASE_ID, RETIRED_RELEASE_IDS
from evaluation_access.server import Config
from isolated_runs.launcher import LaunchError, Launcher, approved_release_id
from scripts import export_dataset


@pytest.mark.parametrize("release_id", sorted(RETIRED_RELEASE_IDS))
def test_retired_dataset_cannot_be_approved_or_served(tmp_path, release_id):
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"dataset_release_id": release_id}))
    dataset = {"source": str(tmp_path), "files": {
        "release.json": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    with pytest.raises(LaunchError, match="withdrawn as invalid"):
        approved_release_id(dataset)
    with pytest.raises(ValueError, match="withdrawn as invalid"):
        Config(state_root=tmp_path / "state", molecules_dir=tmp_path,
               release_id=release_id)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("release_id", sorted(RETIRED_RELEASE_IDS))
def test_stale_run_cannot_resume_even_after_dataset_deleted(tmp_path, monkeypatch, release_id):
    launcher = Launcher(tmp_path)
    control = tmp_path / "old-run" / "control"
    control.mkdir(parents=True)
    (control / "config.json").write_text(json.dumps({"auth": {"kind": "api_key"}}))
    (control / "runtime.json").write_text(json.dumps({"expected_release_id": release_id}))
    monkeypatch.setattr(launcher, "state", lambda name: {"status": "stopped"})
    monkeypatch.setattr(launcher, "preflight", lambda *a: pytest.fail("must reject before side effects"))
    with pytest.raises(LaunchError, match="withdrawn as invalid"):
        launcher.start("old-run", resume=True)


def test_only_current_dataset_can_be_exported():
    assert {profile["release_id"] for profile in export_dataset.SOURCE_PROFILES.values()} == {CURRENT_RELEASE_ID}
