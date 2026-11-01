#!/usr/bin/env bash
# Lightweight watchdog for the long LAFAN1 runs.
#
# It only restarts a run when:
#   * no `sm-train --run-name <run>` process is alive,
#   * the run has not reached TOTAL_ITERATIONS,
#   * a checkpoint exists,
#   * and no `<run>.stop` marker exists.
#
# Start it detached:
#   setsid nohup bash experiment/koopman-world-model/watch_lafan1_runs.sh \
#       > /root/autodl-tmp/logs/watch_lafan1.log 2>&1 &
#
# To stop a run and prevent restart:
#   touch /root/autodl-tmp/logs/<run>.stop
set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_ROOT=${LOG_ROOT:-/root/autodl-tmp/logs}
SCRIPT=${SCRIPT:-experiment/koopman-world-model/run_lafan1_online.sh}
SLEEP=${WATCH_SLEEP:-300}
MAX_RESTARTS=${MAX_RESTARTS:-3}
TOTAL=${TOTAL_ITERATIONS:-320000}
MAX_HOURS=${MAX_HOURS:-24}

STATE_RUN=${STATE_RUN:-lafan1_state_koopman_plan5_fix}
MLP_RUN=${MLP_RUN:-lafan1_mlp_online_plan5}

watch_one() {
    local run=$1
    local kind=$2
    local log="$LOG_ROOT/$run.nohup.log"
    local stop="$LOG_ROOT/$run.stop"
    local done_marker="$LOG_ROOT/$run.done"
    local restart_marker="$LOG_ROOT/$run.restarts"
    local ckpt="$LOG_ROOT/$run/ckpt_latest.pt"

    if [ -f "$stop" ]; then
        return
    fi
    if pgrep -f "sm-train --run-name $run" >/dev/null 2>&1; then
        return
    fi

    local last_it
    last_it=$(grep -oE '^it [0-9]+' "$log" 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -n "${last_it:-}" ] && [ "$last_it" -ge "$TOTAL" ]; then
        touch "$done_marker"
        echo "[watch] $run reached $last_it; marked done"
        return
    fi

    if [ ! -f "$ckpt" ]; then
        echo "[watch] $run is down but has no checkpoint yet ($ckpt); waiting"
        return
    fi

    local restarts=0
    [ -f "$restart_marker" ] && restarts=$(cat "$restart_marker" 2>/dev/null || echo 0)
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "[watch] $run exceeded MAX_RESTARTS=$MAX_RESTARTS; stopping"
        touch "$stop"
        return
    fi
    restarts=$((restarts + 1))
    echo "$restarts" > "$restart_marker"
    echo "[watch] restarting $run ($kind) from $ckpt, attempt $restarts"

    if [ "$kind" = "mlp" ]; then
        setsid nohup env \
            NUM_ENVS=8192 TOTAL_ITERATIONS="$TOTAL" MAX_HOURS="$MAX_HOURS" \
            RUN_NAME="$run" LOG_ROOT="$LOG_ROOT" \
            WM_TYPE=mlp STAB=0 XI_STATE=0 STATE_KOOPMAN=0 \
            RESUME="$ckpt" RESUME_MODE=full \
            bash "$SCRIPT" >> "$LOG_ROOT/$run.resume.log" 2>&1 &
    else
        setsid nohup env \
            NUM_ENVS=8192 TOTAL_ITERATIONS="$TOTAL" MAX_HOURS="$MAX_HOURS" \
            RUN_NAME="$run" LOG_ROOT="$LOG_ROOT" \
            STATE_KOOPMAN=1 XI_STATE=1 STAB=1 \
            RESUME="$ckpt" RESUME_MODE=full \
            bash "$SCRIPT" >> "$LOG_ROOT/$run.resume.log" 2>&1 &
    fi
}

echo "[watch] started $(date -Is); state=$STATE_RUN mlp=$MLP_RUN sleep=${SLEEP}s"
while true; do
    watch_one "$STATE_RUN" state
    watch_one "$MLP_RUN" mlp
    sleep "$SLEEP"
done
