"""Verify and atomically install a prepared release at a new worker-visible path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def verify(root: Path) -> dict:
    manifest = json.loads((root / "release.json").read_text())
    checksum_path = root / manifest["checksums_file"]
    if hashlib.sha256(checksum_path.read_bytes()).hexdigest() != manifest["checksums_sha256"]:
        raise ValueError("Release checksum map mismatch")
    hashes = json.loads(checksum_path.read_text())
    for relative, expected in hashes.items():
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Unsafe release path: {relative}") from exc
        if path.is_symlink():
            raise ValueError(f"Unsafe release path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release payload mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(hashes) | {"release.json", manifest["checksums_file"]}:
        raise ValueError("Release has missing or unexpected files")
    return manifest


def install(source: Path, destination: Path) -> Path:
    source, destination = source.resolve(), destination.absolute()
    manifest = verify(source)
    release_id = manifest["dataset_release_id"]
    if Path(release_id).name != release_id or release_id in {".", ".."}:
        raise ValueError("Invalid release identity")
    if destination.name != release_id:
        raise ValueError("Destination must end in the dataset release ID")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ValueError("Destination cannot be a symlink")
        verify(destination)
        if (source / "release.json").read_bytes() != (destination / "release.json").read_bytes():
            raise ValueError("Existing release identity differs; refusing overwrite")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-staging-", dir=destination.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        verify(staging)
        for path in staging.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        staging.chmod(0o555)
        os.rename(staging, destination)
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
            staging.chmod(0o755)
            shutil.rmtree(staging)
        raise
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    output = install(args.source, args.destination)
    print(json.dumps({"path": str(output), "release": verify(output)["dataset_release_id"],
                      "release_sha256": hashlib.sha256((output / "release.json").read_bytes()).hexdigest()}))
