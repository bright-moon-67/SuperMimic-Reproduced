#!/usr/bin/env bash
# Quick offline parallel validation: MLP 2048^3 vs Koopman on LAFAN1.
#
#   bash experiment/koopman-world-model/run_offline_quick_parallel.sh
#
# This is deliberately a *quick screen*, not the full paper-aligned schedule:
#   - data: 40 LAFAN1 clips, pretrained-policy rollouts, return sampling,
#           8192 envs x 1000 steps (replay capacity 1000 -> 8.19M rows)
#   - MLP:  (2048, 2048, 2048)
#   - Koopman: dz128/r8/256 (0.44M) and dz256/r16/1024 (4.50M)
#   - training: batch 4096 x 1 batch/update, 2000 updates, lr 2e-4, compile default
#
# The full upstream-aligned WM update is batch 4096 x 10; use
# run_paper_aligned.sh for that.  This script answers "does the 2048^3 MLP
# clearly beat Koopman on the same frozen data?" in ~30 min, not hours.
#
# Overrides:
#   DATA=... POLICY=... NUM_ENVS=8192 COLLECT_STEPS=1000
#   UPDATES=2000 EVAL_INTERVAL=500 BATCH_SIZE=4096 BATCHES_PER_UPDATE=1
#   RECOLLECT=1   # rebuild the 40-clip snapshot
#   STAB=0        # disable P2-B for Koopman
#   XI_STATE=1    # augment latent state with xi: z~=[z, xi]
#   WITH_LARGE=0  # skip the 4.50M Koopman run
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-/root/autodl-tmp/data/wm/lafan1_paper_quick.pt}
POLICY=${POLICY:-/root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt}
LOGS=${LOGS:-/root/autodl-tmp/logs/wm}
OUT=${OUT:-experiment/koopman-world-model}

NUM_ENVS=${NUM_ENVS:-8192}
COLLECT_STEPS=${COLLECT_STEPS:-1000}
UPDATES=${UPDATES:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
BATCH_SIZE=${BATCH_SIZE:-4096}
BATCHES_PER_UPDATE=${BATCHES_PER_UPDATE:-1}
STAB=${STAB:-1}
XI_STATE=${XI_STATE:-0}
RECOLLECT=${RECOLLECT:-0}
WITH_LARGE=${WITH_LARGE:-1}

mkdir -p "$LOGS"

if [ ! -f "$DATA" ] || [ "$RECOLLECT" = "1" ]; then
    if [ ! -f "$POLICY" ]; then
        echo "[error] policy checkpoint not found: $POLICY" >&2
        exit 1
    fi
    echo "[collect] 40 LAFAN1 clips, policy rollouts, ${NUM_ENVS} envs x ${COLLECT_STEPS} steps" >&2
    .venv/bin/python experiment/koopman-world-model/collect_wm_data.py \
        --paper \
        --policy-ckpt "$POLICY" \
        --num-envs "$NUM_ENVS" \
        --collect-steps "$COLLECT_STEPS" \
        --sampling-mode return \
        --action-std 0.1 \
        --eval-horizons 8 32 \
        --out "$DATA"
else
    echo "[collect] using existing snapshot $DATA (RECOLLECT=1 to rebuild)" >&2
fi

STAB_ARGS=()
if [ "$STAB" = "1" ]; then
    STAB_ARGS=(
        --a-rho-target 0.99
        --stab-mode power
        --lambda-stab 10.0
        --stab-iters 10
        --stab-batch 256
    )
fi
XI_ARGS=()
if [ "$XI_STATE" = "1" ]; then
    XI_ARGS=(--koopman-use-xi-state)
fi

RUN_PID=""
run() {
    local name=$1
    shift
    echo "[launch] ${name}" >&2
    nohup .venv/bin/python experiment/koopman-world-model/train_wm_only.py "$@" \
        --out "$OUT/wm_quick_${name}.json" \
        --save-wm "$LOGS/quick_${name}.pt" \
        > "$LOGS/quick_${name}.log" 2>&1 &
    RUN_PID=$!
}

pids=()
COMMON=(--protocol a --data-in "$DATA" --batch-size "$BATCH_SIZE" --batches-per-update "$BATCHES_PER_UPDATE" --updates "$UPDATES" --eval-interval "$EVAL_INTERVAL" --compile default)

run mlp --wm-type mlp "${COMMON[@]}" --eval-modes latent --hidden 2048 2048 2048
pids+=("$RUN_PID")

run koopman --wm-type koopman "${COMMON[@]}" --eval-modes latent,reencode \
    --dz 128 --rank 8 --koopman-enc-hidden 256 256 --koopman-dec-hidden 256 256 \
    "${STAB_ARGS[@]}" "${XI_ARGS[@]}"
pids+=("$RUN_PID")

if [ "$WITH_LARGE" = "1" ]; then
    run koopman_large --wm-type koopman "${COMMON[@]}" --eval-modes latent \
        --dz 256 --rank 16 --koopman-enc-hidden 1024 1024 --koopman-dec-hidden 1024 1024 \
        "${STAB_ARGS[@]}" "${XI_ARGS[@]}"
    pids+=("$RUN_PID")
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
echo "[done] status=${status}; logs in ${LOGS}" >&2
exit "$status"
