"""Bounded current-run decision tables and conservative progress projection.

No filesystem access lives here. The operator monitor supplies bytes through
its no-follow run/export view. The record writer defines the only supported
schema; historical paper data is deliberately neither read nor imported.
"""
from __future__ import annotations

from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import io
import math
import re

from scripts.record_decision import ENERGY_MINIMUM, FIELDS, SIGNIFICANT_CHANGE, same_commit

TABLE_NAMES = ("results", "generalizable", "non_generalizable")
TABLE_PATHS = {name: f"workspace/{name}.tsv" for name in TABLE_NAMES}
MAX_TABLE_BYTES = 4 * 1024 * 1024
MAX_TABLE_ROWS = 5000
# A single multiline-free description may exceed csv's default 128 KiB.
# The complete file is bounded before parsing, including any one cell.
csv.field_size_limit(MAX_TABLE_BYTES)
COMMIT = re.compile(r"[a-f0-9]{7,64}\Z")
IDENTITY = re.compile(r"[A-Za-z0-9._:/-]{1,128}\Z")


def parse_table(data: bytes | None, *, availability="unavailable", modified=None, size=None):
    """Return every complete bounded row, retaining raw audit text separately.

    A non-newline-terminated final record may still be being appended and is
    withheld. A malformed interior row or exceeded limit disables the plot:
    silently dropping history could invent a champion trajectory.
    """
    result = {"availability": availability, "columns": [], "rows": [], "raw": None,
              "warnings": [], "plot_safe": False, "partial_final_row": False,
              "modified_at": modified, "size_bytes": size}
    if data is None:
        return result
    if len(data) > MAX_TABLE_BYTES:
        result.update(availability="oversized")
        return result
    try:
        raw = data.decode("utf-8")
    except UnicodeError:
        result.update(availability="malformed", warnings=["Invalid UTF-8; no table or plot is shown."])
        return result
    result["raw"] = raw
    if not raw:
        result.update(availability="empty", plot_safe=True)
        return result
    text = raw
    if not text.endswith("\n"):
        text = text.rpartition("\n")[0] + ("\n" if "\n" in text else "")
        result["partial_final_row"] = True
        result["warnings"].append("Final line is incomplete and withheld until its terminating newline arrives.")
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
        header = next(reader, None)
        if header != list(FIELDS):
            result.update(availability="schema_mismatch" if header else "writing")
            result["warnings"].append("Waiting for the complete current record_decision.py header." if not header
                                      else "Unsupported TSV header; historical or mixed schemas are not plotted.")
            return result
        result["columns"] = header
        for line, cells in enumerate(reader, 2):
            if len(result["rows"]) >= MAX_TABLE_ROWS:
                result.update(availability="row_limit")
                result["warnings"].append(f"More than {MAX_TABLE_ROWS} rows; table view is capped and the plot is disabled.")
                return result
            if (len(cells) != len(header)
                    or any("\n" in cell or "\r" in cell or "\t" in cell for cell in cells)):
                result.update(availability="malformed")
                result["warnings"].append(f"Malformed record at line {line}; the plot is disabled.")
                return result
            result["rows"].append(dict(zip(header, cells)))
    except csv.Error:
        result.update(availability="malformed")
        result["warnings"].append("Invalid TSV quoting; the plot is disabled.")
        return result
    result.update(availability="available", plot_safe=True)
    return result


def summary(table):
    return {key: value for key, value in table.items() if key not in {"rows", "columns", "raw"}} | {
        "row_count": len(table["rows"]), "raw_available": table["raw"] is not None}


def numeric(row, field):
    try:
        value = Decimal(row[field])
        return value if value.is_finite() and math.isfinite(float(value)) else None
    except (KeyError, InvalidOperation, ValueError, OverflowError):
        return None


def integer(row, field):
    value = numeric(row, field)
    return value if value is not None and value == value.to_integral_value() and 0 <= value <= 10**9 else None


def split_metrics(row, split):
    """Validate complete-record metrics before exposing cost or energy recovery."""
    prefix = split + "_"
    steps, energy = numeric(row, prefix + "mean_rel_steps"), numeric(row, prefix + "mean_rel_energy")
    errors = integer(row, prefix + "internal_error_count")
    count = integer(row, prefix + "molecule_count")
    converged = integer(row, prefix + "converged_count")
    fraction = numeric(row, prefix + "converged_fraction")
    retries = integer(row, prefix + "infrastructure_retries")
    if (steps is None or energy is None or errors is None or count is None or count <= 0
            or converged is None or converged > count or fraction is None or not 0 <= fraction <= 1
            or retries is None or row[prefix + "is_valid"] not in {"0", "1"}
            or not IDENTITY.fullmatch(row[prefix + "evaluation_id"])
            or not IDENTITY.fullmatch(row[prefix + "program_id"])):
        raise ValueError("Incomplete or malformed completed evaluation metrics")
    # The evaluator's internal-error penalty is 1000 steps / -1 energy.
    # No large-cost heuristic: a genuine high measured cost remains visible.
    measured = errors == 0 and steps >= 0 and steps != 1000 and energy != -1
    good = measured and row[prefix + "is_valid"] == "1" and energy >= ENERGY_MINIMUM
    return {"cost": float(steps) if measured else None,
            "energy_ratio": float(energy) if measured else None, "valid": good, "errors": int(errors)}


def fingerprint(row):
    return tuple(row[field] for field in FIELDS if field != "timestamp")


def classify(row):
    """Use explicit decisions/reasons, then verify their numerical preconditions."""
    cycle = integer(row, "cycle")
    if (cycle is None or cycle < 1 or row["cycle"] != str(int(cycle))
            or not COMMIT.fullmatch(row["candidate_commit"])
            or not COMMIT.fullmatch(row["anchor_commit"]) or not IDENTITY.fullmatch(row["release_id"])
            or not IDENTITY.fullmatch(row["train_evaluation_id"])
            or not IDENTITY.fullmatch(row["train_program_id"])):
        raise ValueError("Missing or invalid cycle/candidate/release/evaluation provenance")
    if row["valid_program_id"] and row["valid_program_id"] != row["train_program_id"]:
        raise ValueError("Training and validation program identities differ")
    point = {"cycle": int(cycle), "status": "unverified", "train_cost": None, "valid_cost": None,
             "train_mean_rel_energy": None, "valid_mean_rel_energy": None,
             **{key: row[key] for key in ("decision", "candidate_commit", "anchor_commit", "release_id",
                 "train_evaluation_id", "valid_evaluation_id", "timestamp", "description", "validity_reason")}}
    decision, reason = row["decision"], row["validity_reason"]
    if decision == "pending":
        if row["accepted"] != "" or not reason.startswith(("train evaluation pending;", "valid evaluation pending;")):
            raise ValueError("Pending record has inconsistent decision fields")
        point["status"] = "pending"
        # The writer reaches validation-pending only after completed valid
        # training. Show that split's measured energy, without assigning a
        # provisional cost score to the unfinished cycle.
        if reason.startswith("valid evaluation pending;"):
            try:
                train = split_metrics(row, "train")
            except ValueError:
                pass  # Incomplete training evidence remains unavailable.
            else:
                if train["valid"]:
                    point["train_mean_rel_energy"] = train["energy_ratio"]
        return point
    if row["accepted"] != ("1" if decision in {"anchor", "keep"} else "0"):
        raise ValueError("Acceptance flag disagrees with the explicit decision")
    train = split_metrics(row, "train")
    point["train_cost"] = train["cost"]
    point["train_mean_rel_energy"] = train["energy_ratio"]
    validation = split_metrics(row, "valid") if row["valid_evaluation_id"] else None
    if validation:
        point["valid_cost"] = validation["cost"]
        point["valid_mean_rel_energy"] = validation["energy_ratio"]
    if train["errors"] or (validation and validation["errors"]):
        if decision not in {"invalid", "non_generalizable"} or not reason.startswith(("train invalid:", "valid split invalid:")):
            raise ValueError("Internal errors conflict with the recorded decision")
        point.update(status="error", train_cost=None, valid_cost=None)
        return point
    if train["cost"] is None or (validation and validation["cost"] is None):
        raise ValueError("Penalty sentinel or non-measured aggregate is withheld")
    if decision == "anchor":
        if (cycle != 1 or not same_commit(row["candidate_commit"], row["anchor_commit"])
                or not train["valid"] or not validation or not validation["valid"]
                or row["train_improvement_vs_anchor"] or row["valid_improvement_vs_anchor"]
                or reason != "complete valid starting train/valid anchor; no improvement claim"):
            raise ValueError("Starting-anchor evidence is inconsistent")
        point["status"] = "anchor"
        return point
    if decision == "invalid":
        rejected = (reason.startswith("train invalid:") and not train["valid"]) or (
            reason.startswith("valid split invalid:") and validation and not validation["valid"])
        if not rejected:
            raise ValueError("Validity rejection lacks matching invalid split evidence")
        point["status"] = "validity_reject"
        return point
    train_improvement = numeric(row, "train_improvement_vs_anchor")
    valid_improvement = numeric(row, "valid_improvement_vs_anchor")
    if not train["valid"] or train_improvement is None:
        raise ValueError("Completed decision lacks valid training/improvement evidence")
    for split in (("train", "valid") if decision in {"keep", "non_generalizable"} else ("train",)):
        baseline, value = numeric(row, f"anchor_{split}_mean_rel_steps"), numeric(row, f"{split}_mean_rel_steps")
        if (baseline is None or value is None or baseline - value != numeric(row, f"{split}_improvement_vs_anchor")
                or not IDENTITY.fullmatch(row[f"anchor_{split}_evaluation_id"])):
            raise ValueError("Improvement does not match the identified anchor metrics")
    if decision == "discard":
        if reason != "train improvement below 1e-4" or train_improvement >= SIGNIFICANT_CHANGE:
            raise ValueError("No-improvement decision conflicts with its reason or metric")
        point["status"] = "no_improvement"
    elif decision in {"keep", "non_generalizable"}:
        if train_improvement < SIGNIFICANT_CHANGE or not validation or valid_improvement is None:
            raise ValueError("Generalization decision lacks completed split/improvement evidence")
        if decision == "keep":
            if (not validation["valid"] or valid_improvement <= SIGNIFICANT_CHANGE
                    or reason != "train improvement >= 1e-4; valid improvement > 1e-4; both evaluations valid"):
                raise ValueError("Keep conflicts with the explicit split gates")
            point["status"] = "accepted"
        elif reason.startswith("valid split invalid:") and not validation["valid"]:
            point["status"] = "validity_reject"
        elif (reason == "valid improvement is not greater than 1e-4" and validation["valid"]
                and valid_improvement <= SIGNIFICANT_CHANGE):
            point["status"] = "generalization_reject"
        else:
            raise ValueError("Generalization rejection lacks matching reason/evidence")
    else:
        raise ValueError("Unsupported explicit decision")
    return point


def build_progress(tables, *, initial_commit=None):
    """Join exact records, resolve pending appends, and identify champion lineage."""
    warnings = [f"{name}.tsv: {warning}" for name, table in tables.items() for warning in table["warnings"]]
    output = {"points": [], "champions": [], "counts": {}, "warnings": warnings,
              "source": "Current run TSV records only", "state": "empty", "record_count": 0}
    results = tables["results"]
    if results["availability"] == "missing":
        output["message"] = "No recorded AR cycles yet. This run has not written results.tsv; starting-anchor preparation is shown in Run details."
        return output
    if not results["plot_safe"]:
        output.update(state="unavailable", message=f"No progress plot: results.tsv is {results['availability']}.")
        return output
    if not results["rows"]:
        output["message"] = "No recorded AR cycles yet. Starting-anchor preparation is shown in Run details."
        return output
    groups = {}
    output["record_count"] = len(results["rows"])
    for row in results["rows"]:
        groups.setdefault(row["cycle"], []).append(row)
    decisions = {}
    for name in ("generalizable", "non_generalizable"):
        decisions[name] = {}
        if tables[name]["plot_safe"]:
            for row in tables[name]["rows"]:
                decisions[name].setdefault(row["cycle"], []).append(row)
    selected = []
    for cycle, rows in groups.items():
        try:
            # Only append-only pending -> completed resolution is legitimate.
            completed = [row for row in rows if row["decision"] != "pending"]
            if (len({row["candidate_commit"] for row in rows}) != 1
                    or len({row["release_id"] for row in rows}) != 1
                    or len({fingerprint(row) for row in completed}) > 1
                    or (completed and any(row["decision"] == "pending" for row in rows[rows.index(completed[0]) + 1:]))):
                raise ValueError("Conflicting duplicate cycle; all versions withheld")
            row = completed[-1] if completed else rows[-1]
            point = classify(row)
            if (point["status"] == "anchor" and initial_commit is not None
                    and (not isinstance(initial_commit, str) or not COMMIT.fullmatch(initial_commit)
                         or not same_commit(row["candidate_commit"], initial_commit))):
                point.update(status="unverified", train_cost=None, valid_cost=None,
                             train_mean_rel_energy=None, valid_mean_rel_energy=None)
                warnings.append(f"Cycle {cycle}: starting anchor does not match this run's pinned initial candidate.")
            target = ("generalizable" if row["decision"] in {"anchor", "keep"} else
                      "non_generalizable" if row["decision"] == "non_generalizable" else None)
            if target:
                matches = decisions[target].get(cycle, [])
                if not matches or any(fingerprint(match) != fingerprint(row) for match in matches):
                    point.update(status="unverified", train_cost=None, valid_cost=None,
                                 train_mean_rel_energy=None, valid_mean_rel_energy=None)
                    warnings.append(f"Cycle {cycle}: awaiting an identical {target}.tsv record; outcome withheld.")
            selected.append((point, row))
        except ValueError as exc:
            warnings.append(f"Cycle {cycle[:24]}: {exc}.")
    selected.sort(key=lambda item: item[0]["cycle"])
    champion = None
    for point, row in selected:
        if point["status"] in {"accepted", "anchor"}:
            if point["status"] == "accepted":
                anchored = champion is not None and same_commit(row["anchor_commit"], champion["candidate_commit"])
                for split in ("train", "valid"):
                    anchored = (anchored and row[f"anchor_{split}_evaluation_id"] == champion[f"{split}_evaluation_id"]
                                and numeric(row, f"anchor_{split}_mean_rel_steps") == numeric(champion, f"{split}_mean_rel_steps"))
                if not anchored or row["release_id"] != champion["release_id"]:
                    point.update(status="unverified", train_cost=None, valid_cost=None,
                                 train_mean_rel_energy=None, valid_mean_rel_energy=None)
                    warnings.append(f"Cycle {point['cycle']}: champion lineage is incomplete or inconsistent; keep withheld.")
                else:
                    champion = row
                    output["champions"].append(point)
            else:
                champion = row
                output["champions"].append(point)
        output["points"].append(point)
    for name, records in decisions.items():
        if any(cycle not in groups for cycle in records):
            warnings.append(f"{name}.tsv contains orphan cycles absent from results.tsv; they are not plotted.")
    output["counts"] = dict(Counter(point["status"] for point in output["points"]))
    output["state"] = "available" if output["points"] else "unavailable"
    output["message"] = ("Hover or focus a marker for cycle, decision, and evaluation provenance."
                         if output["points"] else "No consistent complete cycle records can be plotted.")
    return output
