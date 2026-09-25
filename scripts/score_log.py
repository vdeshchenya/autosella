#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_last_json(path: Path) -> dict:
    """Read an evaluator JSON document or the last JSON object in an append-only log."""
    contents = path.read_text(encoding="utf-8")
    try:
        document = json.loads(contents)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict):
        return document
    found = None
    for line in contents.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            found = json.loads(line)
    if found is None:
        raise SystemExit(f"No JSON result found in {path}")
    return found


def main() -> None:
    log_path = Path(sys.argv[1] if len(sys.argv) > 1 else "run.log")
    result = load_last_json(log_path)
    print(json.dumps(result, sort_keys=True))
    fields = (
        "status", "evaluation_id", "program_id", "split", "fitness",
        "mean_rel_steps", "mean_rel_energy", "converged_count", "molecule_count",
        "converged_fraction", "internal_error_count", "infrastructure_retries",
        "stop_reason_counts", "is_valid", "validity_reason",
    )
    print(" ".join(f"{name}={json.dumps(result[name], sort_keys=True)}"
                   for name in fields if name in result))


if __name__ == "__main__":
    main()
