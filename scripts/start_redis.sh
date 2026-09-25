#!/usr/bin/env bash
set -Eeuo pipefail
PORT="${1:-6379}"
exec redis-server --port "$PORT" --save "" --appendonly no
