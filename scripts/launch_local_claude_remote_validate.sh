#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT"
exec "${PYTHON_BIN:-python3}" -m isolated_runs.launcher --provider claude "$@"
