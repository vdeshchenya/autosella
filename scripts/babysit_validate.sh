#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <redis_host> <redis_port> <local_redis_port> <num_workers> [mode] [poll_interval] [verbose] [log_dir] [python_bin] [xtb_num_threads] [max_tasks] [task_timeout_seconds] [max_task_rss_gb]"
    echo "  <redis_host>: Redis host. Use localhost when Redis is on this machine."
    echo "  <redis_port>: Redis port on the Redis host."
    echo "  <local_redis_port>: Local Redis port for workers. Use redis_port for local Redis."
    echo "  <num_workers>: Number of worker processes to keep running."
    echo "  [mode]: 'xtb' (the only supported mode; default: xtb)"
    echo "  [poll_interval]: Worker poll interval in seconds (default: 1.0)"
    echo "  [verbose]: Enable verbose output (default: false)"
    echo "  [log_dir]: Directory for log files (default: logs_validate)"
    echo "  [python_bin]: Python executable to use (default: python)"
    echo "  [xtb_num_threads]: Number of xTB threads per worker (default: 1)"
    echo "  [max_tasks]: Respawn each worker after this many tasks (default: 300, 0 = never)"
    echo "  [task_timeout_seconds]: Kill one worker task after this many seconds (default: 2100, 0 = disabled)"
    echo "  [max_task_rss_gb]: Kill one worker task above this process-group RSS (default: 32, 0 = disabled)"
    echo ""
    echo "Local Redis example: $0 localhost 6379 6379 48 xtb 1.0 false logs_validate python 1 300 2100 32"
    echo "Remote Redis example: $0 redis-host 6379 6380 8 xtb 1.0 false logs_validate python 1 300 2100 32"
    exit 1
fi

REMOTE_HOST=$1
REDIS_PORT=$2
LOCAL_REDIS_PORT=$3
NUM_WORKERS=$4
MODE=${5:-xtb}
if [ "$MODE" != "xtb" ]; then
    echo "Only xtb workers are supported by this AR evaluator." >&2
    exit 1
fi
POLL_INTERVAL=${6:-1.0}
VERBOSE=${7:-false}
LOG_DIR=${8:-logs_validate}
PYTHON_BIN=${9:-python}
XTB_NUM_THREADS=${10:-1}
MAX_TASKS=${11:-300}
TASK_TIMEOUT_SECONDS=${12:-${TASK_TIMEOUT_SECONDS:-2100}}
MAX_TASK_RSS_GB=${13:-${MAX_TASK_RSS_GB:-32}}

TUNNEL_STARTED_BY_THIS_SCRIPT=false

is_local_host() {
    case "$1" in
        localhost|127.0.0.1|::1|"$(hostname)"|"$(hostname -f 2>/dev/null || true)")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

has_matching_tunnel() {
    local pids
    pids=$(ss -ltnpH "( sport = :${LOCAL_REDIS_PORT} )" 2>/dev/null | sed -n '/ssh/s/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u)

    for pid in $pids; do
        if [ -r "/proc/$pid/cmdline" ]; then
            local cmd
            cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
            if echo "$cmd" | grep -Eq -- "-L[[:space:]]*${LOCAL_REDIS_PORT}:(localhost|127\\.0\\.0\\.1|\\[::1\\]):${REDIS_PORT}"; then
                if echo "$cmd" | grep -Eq -- "[[:space:]]${REMOTE_HOST}([[:space:]]|$)"; then
                    return 0
                fi
            fi
        fi
    done
    return 1
}

if is_local_host "$REMOTE_HOST"; then
    if [ "$LOCAL_REDIS_PORT" != "$REDIS_PORT" ]; then
        echo "Local Redis requested; ignoring local_redis_port=$LOCAL_REDIS_PORT and using redis_port=$REDIS_PORT."
    fi
    CLIENT_REDIS_HOST="localhost"
    CLIENT_REDIS_PORT="$REDIS_PORT"
else
    echo "Setting up SSH tunnel to $REMOTE_HOST:$REDIS_PORT via local port $LOCAL_REDIS_PORT..."
    if has_matching_tunnel; then
        echo "Found existing SSH tunnel ${LOCAL_REDIS_PORT}->${REMOTE_HOST}:${REDIS_PORT}; use another local port."
        exit 1
    fi

    ssh -f -N -L "$LOCAL_REDIS_PORT:localhost:$REDIS_PORT" "$REMOTE_HOST"
    echo "SSH tunnel established"
    TUNNEL_STARTED_BY_THIS_SCRIPT=true
    CLIENT_REDIS_HOST="localhost"
    CLIENT_REDIS_PORT="$LOCAL_REDIS_PORT"
fi

AVAILABLE_THREADS=$(nproc)

echo "Starting $NUM_WORKERS babysat validation workers"
echo "Server: $CLIENT_REDIS_HOST at port $CLIENT_REDIS_PORT"
echo "Mode: $MODE"
echo "Available CPU threads on this machine: $AVAILABLE_THREADS"
echo "Python executable: $PYTHON_BIN"
echo "xTB threads per worker: $XTB_NUM_THREADS"
echo "Max tasks before worker respawn: $MAX_TASKS"
echo "Per-task timeout: $TASK_TIMEOUT_SECONDS seconds"
echo "Per-task RSS limit: $MAX_TASK_RSS_GB GiB"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN"
    exit 1
fi

cleanup() {
    echo "Stopping validation workers..."
    jobs -pr | xargs -r kill

    if [ "${TUNNEL_STARTED_BY_THIS_SCRIPT:-false}" = "true" ]; then
        echo "Closing SSH tunnel..."
        pkill -f "ssh -f -N -L $LOCAL_REDIS_PORT:localhost:$REDIS_PORT $REMOTE_HOST" || true
    else
        echo "No SSH tunnel was started by this script."
    fi
}
trap cleanup EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$LOG_DIR"

for idx in $(seq 1 "$NUM_WORKERS"); do
    echo "Starting validation worker $idx of $NUM_WORKERS..."
    (
        while true; do
            set +e
            PYTHONPATH=$BASE_DIR "$PYTHON_BIN" -m distributed_validate.worker \
                --redis-host "$CLIENT_REDIS_HOST" \
                --redis-port "$CLIENT_REDIS_PORT" \
                --mode "$MODE" \
                --num-threads "$XTB_NUM_THREADS" \
                --poll-interval "$POLL_INTERVAL" \
                --verbose "$VERBOSE" \
                --log-dir "$LOG_DIR" \
                --worker-name "validate-worker-$idx" \
                --max-tasks "$MAX_TASKS" \
                --task-timeout "$TASK_TIMEOUT_SECONDS" \
                --max-rss-gb "$MAX_TASK_RSS_GB"
            status=$?
            set -e
            echo "worker $idx exited with $status; respawning in 2s"
            sleep 2
        done
    ) &
    sleep 1
done

echo "All validation workers started. Press Ctrl+C to stop all workers."
wait
