#!/usr/bin/env bash
set -Eeuo pipefail

# Operator-only diagnostic entry point. Isolated research uses the gateway
# through scripts/run_remote_experiment.sh; this file is not in its snapshot.

DESCRIPTION="${1:-manual}"
RUN_LOG="${RUN_LOG:-run.log}"
PROGRAM="${PROGRAM:-algo.py}"
SPLIT="${SPLIT:-train}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "$PYTHON_BIN" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [ -x "$HOME/miniconda3/envs/autogeomopt/bin/python" ]; then
        PYTHON_BIN="$HOME/miniconda3/envs/autogeomopt/bin/python"
    elif [ -x "$HOME/miniforge3/envs/autogeomopt/bin/python" ]; then
        PYTHON_BIN="$HOME/miniforge3/envs/autogeomopt/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

{
    printf "== Experiment ==\n"
    printf "timestamp: %s\n" "$(date -Is)"
    printf "description:\n%s\n" "$DESCRIPTION"
    printf "program: %s\n" "$PROGRAM"
    printf "split: %s\n" "$SPLIT"
    printf "redis: %s:%s\n" "$REDIS_HOST" "$REDIS_PORT"
    printf "python: %s\n" "$PYTHON_BIN"
    printf "commit: %s\n" "$(git rev-parse --short HEAD 2>/dev/null || printf unknown)"
    printf "branch: %s\n" "$(git branch --show-current 2>/dev/null || printf unknown)"
    printf "== Validation Output ==\n"
} > "$RUN_LOG"

"$PYTHON_BIN" validate.py \
    --program "$PROGRAM" \
    --split "$SPLIT" \
    --redis-host "$REDIS_HOST" \
    --redis-port "$REDIS_PORT" \
    --run-log "$RUN_LOG"

"$PYTHON_BIN" scripts/score_log.py "$RUN_LOG"
