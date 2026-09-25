"""Hash an explicit reviewed file list; never recursively infer an allowlist."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from isolated_runs.launcher import digest, safe_relative, validate_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--files-from", required=True, type=Path, help="one explicitly approved relative file per line")
    p.add_argument("--dataset", action="store_true")
    args = p.parse_args()
    files = [line.strip() for line in args.files_from.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(files) != len(set(files)):
        p.error("duplicate file in allowlist")
    for name in files:
        safe_relative(name, dataset=args.dataset)
    result = {"source": str(args.source.resolve()), "files": {name: digest(args.source / name) for name in files}}
    validate_manifest(result, dataset=args.dataset)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
