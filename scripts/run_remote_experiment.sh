#!/usr/bin/env bash
set -Eeuo pipefail

# Only candidate source is sent; evaluator, dataset and Redis are server-owned.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
EVALUATION_CLIENT="${EVALUATION_CLIENT:-$ROOT/evaluation_access/client.py}"
PROGRAM="${PROGRAM:-algo.py}"
SPLIT="${SPLIT:-train}"
RUN_LOG="${RUN_LOG:-run.log}"
OUTPUT_JSON="${OUTPUT_JSON:-evaluation-${SPLIT}.json}"
: "${EVALUATION_HOST:?Set the restricted evaluation endpoint}"
: "${EVALUATION_KEY:?Set the isolated run SSH private key path}"
args=(evaluate --program "$PROGRAM" --split "$SPLIT" --output-json "$OUTPUT_JSON")
if [[ -n "${EVALUATION_REQUEST_ID:-}" ]]; then
  args+=(--request-id "$EVALUATION_REQUEST_ID")
fi
if [[ -n "${EVALUATION_STATE_FILE:-}" ]]; then
  args+=(--state-file "$EVALUATION_STATE_FILE")
fi
set +e
"$PYTHON" "$EVALUATION_CLIENT" "${args[@]}" | tee -a "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
