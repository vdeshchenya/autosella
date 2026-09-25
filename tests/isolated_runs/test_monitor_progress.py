from __future__ import annotations

import csv
import http.client
import io
import json
import threading

import pytest

from isolated_runs.monitor import MAX_TABLE_BYTES, MonitorError, Server
from isolated_runs.monitor_progress import TABLE_NAMES, build_progress, parse_table
from scripts.record_decision import FIELDS
from monitor_fixture import fixture_records
from test_monitor import docker_for, finished_export, monitor_for, run, write


def tsv(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def tables(rows=None):
    rows = fixture_records() if rows is None else rows
    return {name: parse_table(tsv([row for row in rows if name == "results"
        or (name == "generalizable" and row["decision"] in {"anchor", "keep"})
        or (name == "non_generalizable" and row["decision"] == "non_generalizable")])) for name in TABLE_NAMES}


def test_current_schema_outcomes_and_anchor_separation():
    p = build_progress(tables())
    assert p["counts"] == {"anchor": 1, "accepted": 2, "no_improvement": 1, "validity_reject": 2,
                           "generalization_reject": 1, "error": 1, "pending": 1}
    assert [point["cycle"] for point in p["champions"]] == [1, 2, 8]
    assert p["warnings"] == []
    by_cycle = {point["cycle"]: point for point in p["points"]}
    assert by_cycle[6]["status"] == "validity_reject"  # non_generalizable, but energy gate failed.
    assert by_cycle[5]["status"] == "generalization_reject"
    assert by_cycle[5]["valid_cost"] == .97  # Measured validation remains available on rejection.
    assert by_cycle[6]["valid_cost"] == .9
    assert by_cycle[3]["valid_cost"] is None and by_cycle[3]["valid_evaluation_id"] == ""
    assert by_cycle[7]["train_cost"] is None and by_cycle[7]["valid_cost"] is None
    assert by_cycle[9]["train_cost"] is None and by_cycle[9]["valid_cost"] is None
    assert by_cycle[1]["train_mean_rel_energy"] == by_cycle[1]["valid_mean_rel_energy"] == 1
    assert by_cycle[3]["train_mean_rel_energy"] == 1 and by_cycle[3]["valid_mean_rel_energy"] is None
    assert by_cycle[4]["train_mean_rel_energy"] == .97  # Measured energy-gate rejection.
    assert by_cycle[6]["train_mean_rel_energy"] == 1 and by_cycle[6]["valid_mean_rel_energy"] == .95
    assert by_cycle[7]["train_mean_rel_energy"] is None and by_cycle[7]["valid_mean_rel_energy"] is None
    assert by_cycle[9]["train_mean_rel_energy"] == 1 and by_cycle[9]["valid_mean_rel_energy"] is None
    assert all(point["train_cost"] != 1000 and point["valid_cost"] != -1 for point in p["points"])


def test_validation_internal_error_is_top_tick_not_generalization_rejection():
    rows = fixture_records()
    rows[5].update(valid_internal_error_count=1, valid_mean_rel_steps=1000, valid_mean_rel_energy=-1,
                   valid_validity_reason="internal_error", validity_reason="valid split invalid: internal_error")
    p = build_progress(tables(rows))
    point = next(point for point in p["points"] if point["cycle"] == 6)
    assert point["status"] == "error"
    assert point["train_mean_rel_energy"] == 1  # Completed error-free training remains measured.
    assert point["valid_mean_rel_energy"] is None


@pytest.mark.parametrize("change", [
    {"validity_reason": "train evaluation pending; unscored"},
    {"train_mean_rel_energy": ""},
    {"train_mean_rel_energy": -1},
    {"train_internal_error_count": 1},
    {"train_molecule_count": 0},
])
def test_pending_energy_requires_completed_valid_training_evidence(change):
    rows = fixture_records()
    rows[8].update(change)
    p = build_progress(tables(rows))
    point = next(point for point in p["points"] if point["cycle"] == 9)
    assert point["status"] == "pending"
    assert point["train_cost"] is None and point["valid_cost"] is None
    assert point["train_mean_rel_energy"] is None and point["valid_mean_rel_energy"] is None


def test_no_cost_heuristic_and_sentinel_with_no_error_evidence_is_withheld():
    rows = fixture_records()
    rows[2].update(train_mean_rel_steps=125, train_improvement_vs_anchor="-124.08")
    p = build_progress(tables(rows))
    assert next(point for point in p["points"] if point["cycle"] == 3)["train_cost"] == 125
    rows[2].update(train_mean_rel_steps=1000, train_improvement_vs_anchor="-999.08")
    p = build_progress(tables(rows))
    assert not any(point["cycle"] == 3 for point in p["points"])
    assert any("sentinel" in warning for warning in p["warnings"])


def test_pending_append_resolves_once_and_identical_completed_duplicate_is_idempotent():
    rows = fixture_records()
    pending = {**rows[1], "decision": "pending", "accepted": "", "validity_reason": "valid evaluation pending; provisional train result is unscored"}
    data = tables([rows[0], pending, rows[1], rows[1], *rows[2:]])
    p = build_progress(data)
    assert len(p["points"]) == 9
    assert p["counts"]["accepted"] == 2
    assert p["record_count"] == 11


@pytest.mark.parametrize("change", [{"candidate_commit": "4" * 40}, {"release_id": "wrong-release"},
                                   {"description": "contradictory completed record"}])
def test_conflicting_duplicate_cycle_never_silently_overwrites_history(change):
    rows = fixture_records()
    rows.insert(2, {**rows[1], **change})
    p = build_progress(tables(rows))
    assert not any(point["cycle"] == 2 for point in p["points"])
    assert any("Conflicting duplicate" in warning for warning in p["warnings"])
    assert p["counts"].get("accepted", 0) == 0  # Later keep no longer has a known lineage.


def test_pending_after_completed_is_contradictory_not_reversion():
    rows = fixture_records()
    rows.insert(2, {**rows[1], "decision": "pending", "accepted": "", "validity_reason": "valid evaluation pending; unscored"})
    p = build_progress(tables(rows))
    assert not any(point["cycle"] == 2 for point in p["points"])


@pytest.mark.parametrize("cycle", ["01", "1.0", "1e0", "nan"])
def test_noncanonical_cycle_numbers_cannot_create_duplicate_plot_positions(cycle):
    rows = fixture_records()
    rows.append({**rows[0], "cycle": cycle})
    p = build_progress(tables(rows))
    assert len(p["points"]) == 9 and len(p["champions"]) == 3
    assert p["warnings"]


@pytest.mark.parametrize("field", ["candidate_commit", "release_id", "train_evaluation_id", "valid_evaluation_id",
                                  "train_program_id", "description"])
def test_generalizable_join_requires_identical_provenance_not_just_cycle(field):
    data = tables()
    data["generalizable"]["rows"][1][field] = "changed-value"
    p = build_progress(data)
    point = next(point for point in p["points"] if point["cycle"] == 2)
    assert point["status"] == "unverified"
    assert point["train_mean_rel_energy"] is None and point["valid_mean_rel_energy"] is None
    assert [point["cycle"] for point in p["champions"]] == [1]


@pytest.mark.parametrize("field,value", [("anchor_commit", "9" * 40), ("anchor_train_evaluation_id", "other-evaluation"),
                                       ("anchor_valid_mean_rel_steps", "1.1"), ("release_id", "other-release"),
                                       ("train_improvement_vs_anchor", "0.7")])
def test_keeps_require_anchor_lineage_and_consistent_improvement(field, value):
    rows = fixture_records()
    rows[1][field] = value
    p = build_progress(tables(rows))
    assert p["counts"].get("accepted", 0) == 0
    assert len(p["champions"]) == 1


def test_orphan_decisions_are_not_imported_as_run_history():
    data = tables()
    data["results"] = parse_table(tsv([]))
    p = build_progress(data)
    assert p["state"] == "empty" and p["points"] == [] and p["champions"] == []


def test_starting_anchor_must_match_the_host_pinned_run_candidate():
    p = build_progress(tables(), initial_commit="9" * 40)
    assert p["champions"] == [] and p["counts"].get("accepted", 0) == 0
    assert p["points"][0]["status"] == "unverified"
    assert p["points"][0]["train_mean_rel_energy"] is None
    assert p["points"][0]["valid_mean_rel_energy"] is None
    assert any("pinned initial candidate" in warning for warning in p["warnings"])


def test_missing_empty_old_schema_and_malformed_are_honest_states():
    for payload, state in [(None, "empty"), (b"", "empty"), (tsv([]), "empty"),
                            (b"cycle\tmean_rel_steps\taccepted\n1\t1\t1\n", "unavailable"),
                            (tsv(fixture_records()) + b"broken\trow\n", "unavailable")]:
        data = tables()
        data["results"] = parse_table(payload, availability="missing")
        p = build_progress(data)
        assert p["state"] == state and p["points"] == [] and p["champions"] == []


def test_partial_appended_row_waits_for_newline_and_rows_are_never_silently_truncated(monkeypatch):
    rows = fixture_records()
    data = tables()
    data["results"] = parse_table(tsv(rows)[:-10])
    p = build_progress(data)
    assert len(p["points"]) == 8
    assert data["results"]["partial_final_row"] is True
    assert any("Final line" in warning for warning in p["warnings"])
    import isolated_runs.monitor_progress as module
    monkeypatch.setattr(module, "MAX_TABLE_ROWS", 3)
    limited = parse_table(tsv(rows))
    assert limited["availability"] == "row_limit" and len(limited["rows"]) == 3
    assert not limited["plot_safe"]


def test_live_table_all_columns_no_tail_and_redaction(run):
    root, path, state = run
    rows = fixture_records()
    secret = "exact-provider-secret-in-table-1234567"
    rows[1]["description"] = '<script>alert("xss")</script> ' + secret
    write(path, "workspace/results.tsv", tsv(rows).decode())
    write(path, "secrets/provider", secret)
    with monitor_for(root, docker_for(state)) as monitor:
        table = monitor.table("trial-a", "results")
        assert table["columns"] == list(FIELDS) and len(table["rows"]) == 9
        assert secret not in json.dumps(table) and "[REDACTED]" in table["raw"]
        assert "<script>" in table["rows"][1]["description"]  # UI must use textContent, never HTML.
        for key in ("../secrets/provider", "/etc/passwd", "workspace/results.tsv"):
            with pytest.raises(MonitorError):
                monitor.table("trial-a", key)


def test_unsafe_table_and_byte_limit_never_become_partial_history(run, tmp_path):
    root, path, state = run
    outside = tmp_path / "outside.tsv"
    outside.write_bytes(tsv(fixture_records()))
    target = path / "workspace/results.tsv"
    target.symlink_to(outside)
    with monitor_for(root, docker_for(state)) as monitor:
        assert monitor.table("trial-a", "results")["raw"] is None
        assert monitor.detail("trial-a")["research_progress"]["points"] == []
        target.unlink()
        target.write_bytes(tsv(fixture_records()) + b"x" * MAX_TABLE_BYTES)
        result = monitor.table("trial-a", "results")
        assert result["availability"] == "oversized" and result["raw"] is None
        assert monitor.detail("trial-a")["research_progress"]["points"] == []


def test_finished_tables_use_full_verified_export_and_never_raw_workspace(run, monkeypatch):
    root, path, state = run
    rows = fixture_records()
    rows[2]["description"] = "synthetic large description " * 12000  # Exceeds the old 256 KiB log-tail limit.
    members = [(f"workspace/{name}.tsv", table["raw"]) for name, table in tables(rows).items()]
    finished_export(run, members)
    write(path, "workspace/results.tsv", "unsanitized post-finish raw workspace")
    import isolated_runs.monitor as module
    monkeypatch.setattr(module, "secrets_for", lambda *args: pytest.fail("Finished reads must not access credentials"))
    with monitor_for(root, docker_for(state)) as monitor:
        table = monitor.table("trial-a", "results")
        assert table["availability"] == "available" and len(table["rows"]) == 9
        assert "unsanitized" not in table["raw"]
        p = monitor.detail("trial-a")["research_progress"]
        assert p["counts"]["accepted"] == 2
        assert "Verified sanitized export" in p["source"]


@pytest.mark.parametrize("problem", ["wrong_hash", "missing_member", "oversized"])
def test_finished_missing_unverified_or_oversized_table_does_not_fall_back(run, problem):
    root, path, state = run
    raw = tsv(fixture_records())
    members = [] if problem == "missing_member" else [("workspace/results.tsv", raw if problem != "oversized" else raw + b"x" * MAX_TABLE_BYTES)]
    finished_export(run, members)
    write(path, "workspace/results.tsv", raw.decode())
    if problem == "wrong_hash":
        state["archive"]["sha256"] = "0" * 64
        write(path, "control/state.json", state)
    with monitor_for(root, docker_for(state)) as monitor:
        assert monitor.detail("trial-a")["research_progress"]["points"] == []
        if problem == "wrong_hash":
            with pytest.raises(MonitorError):
                monitor.table("trial-a", "results")
        else:
            assert monitor.table("trial-a", "results")["raw"] is None


def test_table_http_routes_preserve_origin_csp_and_read_only_security(run):
    root, path, state = run
    raw = tsv(fixture_records()).decode()
    write(path, "workspace/results.tsv", raw)
    with monitor_for(root, docker_for(state)) as monitor:
        server = Server(monitor, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(route, headers=None, method="GET"):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                connection.request(method, route, headers=headers or {})
                response = connection.getresponse()
                result = response.status, dict(response.getheaders()), response.read().decode()
                connection.close()
                return result
            code, headers, body = request("/table?name=trial-a&table=results&download=1")
            assert code == 200 and body == raw
            assert headers["Content-Disposition"] == 'attachment; filename="results.tsv"'
            assert "default-src 'none'" in headers["Content-Security-Policy"]
            assert headers["X-Content-Type-Options"] == "nosniff"
            assert request("/api/table?name=trial-a&table=results", {"Origin": "https://evil.example"})[0] == 403
            assert request("/table?name=trial-a&table=results", {"Host": "evil.example"})[0] == 403
            assert request("/table?name=trial-a&table=../../secrets/provider")[0] == 404
            assert request("/api/table?name=trial-a&table=results", method="POST")[0] == 405
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
    assert (path / "workspace/results.tsv").read_text() == raw
