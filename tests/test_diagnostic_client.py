"""Offline read-only diagnostics retrieval and hostile-response checks."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation_access.client import (
    Client, MAX_DIAGNOSTIC_BYTES, MAX_DIAGNOSTIC_CHUNK_BYTES,
    MAX_DIAGNOSTIC_RESPONSE_BYTES, TransportError, diagnostics, main,
)


IDENTITY = {"run_id": "run-a", "evaluation_id": "eval-a", "task_id": "task-a",
            "source_sha256": "a" * 64, "mol_name": "molecule-α"}
PROVENANCE = {**{key: IDENTITY[key] for key in ("run_id", "evaluation_id", "source_sha256")},
              "release_id": "fixture-release", "dataset_release_id": "fixture-release"}


class MockClient:
    expected_release_id = "fixture-release"

    def __init__(self, *, bundle=None, raw=None, chunk_size=45000, mutate=None):
        self.bundle = bundle or {"schema_version": 1, "identity": deepcopy(IDENTITY),
                                 "trace": {"truncated": True, "final_positions": [[1, 2, 3]],
                                           "note": "αβγ" * 16000}}
        self.raw = raw if raw is not None else json.dumps(self.bundle, ensure_ascii=False).encode("utf-8")
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.chunk_size = chunk_size
        self.mutate = mutate
        self.requests = []

    def entry(self):
        return {"artifact_id": self.digest, "identity": deepcopy(IDENTITY),
                "size_bytes": len(self.raw), "summary": {"final_positions": [[1, 2, 3]]},
                "truncated": True}

    def request(self, request):
        self.requests.append(deepcopy(request))
        assert request["method"] == "diagnostics"
        assert set(request) <= {"method", "evaluation_id", "molecule", "artifact_id", "offset", "limit"}
        assert request["evaluation_id"] == "eval-a"
        if "artifact_id" in request:
            assert request["artifact_id"] == self.digest
            assert request["molecule"] == IDENTITY["mol_name"]
            assert request["limit"] <= MAX_DIAGNOSTIC_CHUNK_BYTES
            start = request["offset"]
            chunk = self.raw[start:start + min(self.chunk_size, request["limit"])]
            end = start + len(chunk)
            result = {**PROVENANCE, "artifact_id": self.digest, "sha256": self.digest,
                      "identity": deepcopy(IDENTITY), "offset": start, "next_offset": end,
                      "eof": end == len(self.raw), "encoding": "base64",
                      "data": base64.b64encode(chunk).decode("ascii"), "size_bytes": len(self.raw)}
        else:
            offset = request["offset"]
            result = {**PROVENANCE, "schema_version": 1, "artifacts": [self.entry()],
                      "next_offset": offset + 1, "eof": True, "total": offset + 1}
        if self.mutate:
            self.mutate(result, len(self.requests))
        return result


def download(client, tmp_path, **kwargs):
    return diagnostics(client, evaluation_id="eval-a", molecule=IDENTITY["mol_name"],
                       artifact_id=client.digest, output_dir=tmp_path / "diagnostics", **kwargs)


def test_manifest_is_one_bounded_read_only_page_and_summary_excludes_geometry(tmp_path):
    client = MockClient()
    summary = diagnostics(client, evaluation_id="eval-a", output_dir=tmp_path / "diagnostics")
    assert client.requests == [{"method": "diagnostics", "evaluation_id": "eval-a", "offset": 0, "limit": 32}]
    path = tmp_path / "diagnostics" / "eval-a" / "manifest.json"
    assert summary["diagnostic_manifest"] == str(path)
    assert json.loads(path.read_text())["artifacts"] == [client.entry()]
    assert "final_positions" not in json.dumps(summary)
    assert summary["truncated"] is True
    assert summary["omitted_from_page"] == 0
    assert "absence of failures" in summary["missing_capture"]


def test_manifest_explicit_paging_preserves_first_page(tmp_path):
    client = MockClient()
    root = tmp_path / "diagnostics"
    first = diagnostics(client, evaluation_id="eval-a", output_dir=root)
    first_bytes = Path(first["diagnostic_manifest"]).read_bytes()
    second = diagnostics(client, evaluation_id="eval-a", output_dir=root, offset=32, limit=64,
                         molecule=IDENTITY["mol_name"])
    assert second["next_offset"] == 33
    assert second["omitted_from_page"] == 32
    assert second["diagnostic_manifest"].endswith("manifest-offset-32.json")
    assert Path(first["diagnostic_manifest"]).read_bytes() == first_bytes
    assert client.requests[-1]["limit"] == 64
    assert client.requests[-1]["molecule"] == IDENTITY["mol_name"]


@pytest.mark.parametrize("offset", [0, 32])
def test_empty_manifest_explicitly_preserves_unknown_capture_evidence(tmp_path, offset):
    client = MockClient(mutate=lambda result, count:
                        result.update(artifacts=[], next_offset=0, total=0, eof=True))
    summary = diagnostics(client, evaluation_id="eval-a", output_dir=tmp_path, offset=offset)
    assert summary["total_artifacts"] == 0
    assert summary["capture_evidence"] == "not_recorded"
    assert "absence of failures" in summary["missing_capture"]


def test_manifest_surfaces_explicit_missing_receipt_without_geometry(tmp_path):
    receipt = {"mol_name": IDENTITY["mol_name"], "task_id": IDENTITY["task_id"],
               "availability": "missing", "reason": "capture_not_available",
               "final_positions": [[1, 2, 3]]}
    client = MockClient(mutate=lambda result, count: result.update(
        capture_evidence="available", capture_status_counts={"missing": 1}, capture_receipt=receipt))
    summary = diagnostics(client, evaluation_id="eval-a", molecule=IDENTITY["mol_name"], output_dir=tmp_path)
    assert summary["capture_status_counts"] == {"missing": 1}
    assert summary["capture_evidence"] == "available"
    assert summary["capture_receipt"]["reason"] == "capture_not_available"
    assert summary["missing_capture"] is True
    assert "final_positions" not in json.dumps(summary)


def test_bundle_summary_exposes_truncation_omissions_and_incomplete_capture(tmp_path):
    trace = {"attempted_calls": 301, "completed_calls": 300, "capture_incomplete": True,
             "geometry_omitted_atom_limit": False, "truncation": {
                 "older_completed_frames_not_retained": 292, "older_scalar_rows_not_retained": 101,
                 "frames_dropped_for_size": 1, "attempted_input_omitted_for_size": True},
             "exception": {"traceback_truncated": True, "traceback": "must not print"},
             "invalid_input": {"truncated": True, "values_flat": [1, 2, 3]}}
    client = MockClient(bundle={"schema_version": 1, "identity": IDENTITY, "trace": trace})
    summary = download(client, tmp_path)
    assert summary["truncated"] is True
    assert summary["missing_capture"] is True
    assert summary["omitted"]["older_completed_frames_not_retained"] == 292
    assert summary["omitted"]["invalid_input_truncated"] is True
    assert summary["omitted"]["traceback_truncated"] is True
    assert summary["summary"]["attempted_calls"] == 301
    assert "must not print" not in json.dumps(summary)
    assert "values_flat" not in json.dumps(summary)


def test_complete_bundle_uses_continuous_byte_offsets_and_preserves_unicode(tmp_path):
    client = MockClient()
    summary = download(client, tmp_path)
    assert [request["offset"] for request in client.requests] == [0, 45000, 90000]
    artifact = Path(summary["diagnostic_artifact"])
    assert artifact.read_bytes() == client.raw
    assert json.loads(artifact.read_bytes()) == client.bundle
    assert summary["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert summary["size_bytes"] == len(client.raw)
    assert "final_positions" not in json.dumps(summary)
    assert not list(artifact.parent.glob("*.tmp"))
    assert artifact.stat().st_mode & 0o777 == 0o600


def test_chunk_boundary_may_split_a_utf8_codepoint(tmp_path):
    client = MockClient(chunk_size=127)
    download(client, tmp_path)
    assert any(client.raw[request["offset"]] & 0xc0 == 0x80 for request in client.requests[1:])


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(run_id="run-other"),
    lambda r: r.update(evaluation_id="eval-other"),
    lambda r: r.update(source_sha256="b" * 64),
    lambda r: r.update(release_id="other", dataset_release_id="other"),
    lambda r: r.update(dataset_release_id="other"),
    lambda r: r["identity"].update(task_id="task-other"),
    lambda r: r["identity"].update(mol_name="other"),
    lambda r: r["identity"].update(source_sha256="b" * 64),
    lambda r: r.update(artifact_id="b" * 64),
    lambda r: r.update(sha256="b" * 64),
    lambda r: r.update(offset=r["offset"] + 1),
    lambda r: r.update(offset=float(r["offset"])),
    lambda r: r.update(next_offset=r["next_offset"] + 1),
    lambda r: r.update(eof=True),
    lambda r: r.update(eof=0),
    lambda r: r.update(size_bytes=r["size_bytes"] + 1),
    lambda r: r.update(size_bytes=MAX_DIAGNOSTIC_BYTES + 1),
    lambda r: r.update(data="!invalid-base64!"),
    lambda r: r.update(data=""),
    lambda r: r.update(encoding="utf-8"),
    lambda r: r.update(data=base64.b64encode(b"x" * 45000).decode()),
])
def test_hostile_second_chunk_never_publishes_or_overwrites_artifact(tmp_path, mutation):
    client = MockClient(mutate=lambda response, count: mutation(response) if count == 2 else None)
    artifact = tmp_path / "diagnostics" / "eval-a" / f"{client.digest}.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"previous verified artifact")
    with pytest.raises(ValueError):
        download(client, tmp_path)
    assert artifact.read_bytes() == b"previous verified artifact"
    assert list(artifact.parent.iterdir()) == [artifact]


@pytest.mark.parametrize("raw", [b"not JSON", b'{"schema_version":1,"identity":{},"trace":{}}',
                                 b'{"schema_version":1,"schema_version":1}', b'{"x": NaN}',
                                 b'"\xff"'])
def test_hash_valid_invalid_json_is_not_saved(tmp_path, raw):
    client = MockClient(raw=raw)
    with pytest.raises(ValueError):
        download(client, tmp_path)
    assert not (tmp_path / "diagnostics").exists()


def test_bundle_identity_must_equal_chunk_identity(tmp_path):
    bundle = {"schema_version": 1, "identity": {**IDENTITY, "task_id": "other"}, "trace": {}}
    client = MockClient(bundle=bundle)
    with pytest.raises(ValueError, match="identity"):
        download(client, tmp_path)
    assert not (tmp_path / "diagnostics").exists()


@pytest.mark.parametrize("kwargs", [
    {"evaluation_id": "../escape"}, {"evaluation_id": "/absolute"}, {"evaluation_id": ""},
    {"evaluation_id": ".."}, {"offset": -1}, {"offset": True}, {"offset": 2 ** 31},
    {"limit": 0}, {"limit": 65}, {"limit": True}, {"molecule": "bad\nname"},
    {"artifact_id": "../bad"}, {"artifact_id": "a" * 64},
])
def test_invalid_input_never_sends_a_request(tmp_path, kwargs):
    client = MockClient()
    with pytest.raises(ValueError):
        diagnostics(client, output_dir=tmp_path, **{"evaluation_id": "eval-a", **kwargs})
    assert not client.requests


@pytest.mark.parametrize("kwargs", [{"offset": 1}, {"limit": 64}])
def test_artifact_download_does_not_silently_apply_manifest_paging(tmp_path, kwargs):
    client = MockClient()
    with pytest.raises(ValueError, match="manifest"):
        download(client, tmp_path, **kwargs)
    assert not client.requests


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(artifacts=r["artifacts"] * 33),
    lambda r: r.update(artifacts=r["artifacts"] * 2, next_offset=2, total=2),
    lambda r: r.update(next_offset=0),
    lambda r: r.update(eof=False),
    lambda r: r.update(total=-1),
    lambda r: r.update(schema_version=2),
    lambda r: r.update(artifacts=[]),
    lambda r: r["artifacts"][0].update(artifact_id="../bad"),
    lambda r: r["artifacts"][0].update(size_bytes=MAX_DIAGNOSTIC_BYTES + 1),
    lambda r: r["artifacts"][0]["identity"].update(evaluation_id="other"),
    lambda r: r["artifacts"][0].update(truncated="false"),
    lambda r: r.update(extra="x" * MAX_DIAGNOSTIC_RESPONSE_BYTES),
    lambda r: r.update(capture_evidence="invented"),
    lambda r: r.update(capture_status_counts={"missing": -1}),
    lambda r: r.update(capture_receipt={"mol_name": "other", "availability": "missing"}),
])
def test_hostile_manifest_is_not_saved(tmp_path, mutation):
    client = MockClient(mutate=lambda response, count: mutation(response))
    with pytest.raises(ValueError):
        diagnostics(client, evaluation_id="eval-a", output_dir=tmp_path / "diagnostics")
    assert not (tmp_path / "diagnostics").exists()


@pytest.mark.parametrize("target", ["root", "evaluation", "artifact", "ancestor"])
def test_output_symlinks_cannot_write_outside_selected_root(tmp_path, target):
    client = MockClient()
    root = tmp_path / "diagnostics"
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "root":
        root.symlink_to(outside, target_is_directory=True)
    elif target == "evaluation":
        root.mkdir()
        (root / "eval-a").symlink_to(outside, target_is_directory=True)
    elif target == "ancestor":
        parent = tmp_path / "alias"
        parent.symlink_to(outside, target_is_directory=True)
        root = parent / "diagnostics"
    else:
        (root / "eval-a").mkdir(parents=True)
        existing = outside / "receipt.json"
        existing.write_bytes(b"preserve receipt")
        (root / "eval-a" / f"{client.digest}.json").symlink_to(existing)
    before = [(str(path), path.read_bytes()) for path in outside.iterdir()]
    with pytest.raises((ValueError, OSError)):
        diagnostics(client, evaluation_id="eval-a", molecule=IDENTITY["mol_name"],
                    artifact_id=client.digest, output_dir=root)
    assert [(str(path), path.read_bytes()) for path in outside.iterdir()] == before


def test_manifest_symlink_is_refused(tmp_path):
    client = MockClient()
    root = tmp_path / "diagnostics"
    (root / "eval-a").mkdir(parents=True)
    receipt = tmp_path / "evaluation.json"
    receipt.write_bytes(b"preserve receipt")
    (root / "eval-a" / "manifest.json").symlink_to(receipt)
    with pytest.raises(ValueError, match="symlink"):
        diagnostics(client, evaluation_id="eval-a", output_dir=root)
    assert receipt.read_bytes() == b"preserve receipt"


def test_failed_atomic_rename_removes_temporary_file(tmp_path, monkeypatch):
    client = MockClient()
    def failed(*args, **kwargs):
        raise OSError("fixture replace failed")
    monkeypatch.setattr("evaluation_access.client.os.replace", failed)
    with pytest.raises(OSError, match="replace failed"):
        download(client, tmp_path)
    assert not list((tmp_path / "diagnostics" / "eval-a").iterdir())


def test_diagnostic_transport_has_smaller_response_bound(monkeypatch):
    response = json.dumps({"ok": True, "result": {"padding": "x" * MAX_DIAGNOSTIC_RESPONSE_BYTES}}).encode()
    monkeypatch.setattr("evaluation_access.client.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=response))
    with pytest.raises(TransportError, match="size limit"):
        Client(["offline"]).request({"method": "diagnostics", "evaluation_id": "eval-a"})


def test_decoded_chunk_is_bounded_even_when_response_envelope_fits(tmp_path):
    client = MockClient(mutate=lambda result, count: result.update(
        data=base64.b64encode(b"x" * (MAX_DIAGNOSTIC_CHUNK_BYTES + 1)).decode(),
        next_offset=MAX_DIAGNOSTIC_CHUNK_BYTES + 1))
    with pytest.raises(ValueError, match="chunk offsets"):
        download(client, tmp_path)
    assert not (tmp_path / "diagnostics").exists()


def test_cli_manifest_is_read_only_and_prints_summary(tmp_path, monkeypatch, capsys):
    client = MockClient()
    monkeypatch.setattr("evaluation_access.client.ssh_command", lambda *args, **kwargs: ["offline"])
    monkeypatch.setattr("evaluation_access.client.Client", lambda *args, **kwargs: client)
    receipt, state = tmp_path / "result.json", tmp_path / "client-state.json"
    receipt.write_bytes(b"existing receipt")
    state.write_bytes(b"existing state")
    assert main(["diagnostics", "--evaluation-id", "eval-a", "--state-file", str(state),
                 "--output-dir", str(tmp_path / "diagnostics")]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "final_positions" not in captured.out
    assert json.loads(captured.out)["method"] == "diagnostics"
    assert receipt.read_bytes() == b"existing receipt"
    assert state.read_bytes() == b"existing state"


@pytest.mark.parametrize("error", [ValueError("invalid fixture"), TransportError("offline fixture")])
def test_cli_diagnostic_failures_preserve_evaluation_receipts(tmp_path, monkeypatch, capsys, error):
    receipt = tmp_path / "result.json"
    receipt.write_bytes(b"existing receipt")
    monkeypatch.setattr("evaluation_access.client.ssh_command", lambda *args, **kwargs: ["offline"])
    def failed(*args, **kwargs):
        raise error
    monkeypatch.setattr("evaluation_access.client.Client.request", failed)
    assert main(["diagnostics", "--evaluation-id", "eval-a", "--output-dir", str(tmp_path / "diagnostics")]) in (1, 75)
    assert receipt.read_bytes() == b"existing receipt"
    assert not (tmp_path / "diagnostics").exists()
    assert capsys.readouterr().out == ""
    assert main(["diagnostics", "--evaluation-id", "eval-a", "--output-json", str(receipt)]) == 1
    assert receipt.read_bytes() == b"existing receipt"
