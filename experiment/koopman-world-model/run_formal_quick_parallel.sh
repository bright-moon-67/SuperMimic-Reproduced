#!/usr/bin/env bash
# Fixed-data capacity screen for the plan6 questions.
#
# All runs use the same frozen snapshot and the same fixed a_8/a_32 evaluation
# windows.  This is intentionally still a quick screen (2000 updates, one
# minibatch per update); use run_paper_aligned.sh for the full WM budget.
#
#   bash experiment/koopman-world-model/run_formal_quick_parallel.sh
#
# Runs:
#   mlp_512          ~0.6M MLP, capacity near the small Koopman
#   mlp_1280         ~3.4M MLP, capacity near the large Koopman
#   state_small      state-augmented Koopman, dz128/r8, no true-rec loss
#   state_small_rec  same + L_rec_true=0.1
#   state_large_rec  state-augmented Koopman, dz256/r16, L_rec_true=0.1
#
# The 2048^3 MLP and classic Koopman baselines are already recorded in
# RESULTS.md; rerun them by setting INCLUDE_BASELINES=1.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-/root/autodl-tmp/data/wm/lafan1_paper_quick.pt}
LOGS=${LOGS:-/root/autodl-tmp/logs/formal_quick}
OUT=${OUT:-experiment/koopman-world-model/formal_quick}
UPDATES=${UPDATES:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
BATCH_SIZE=${BATCH_SIZE:-4096}
BATCHES_PER_UPDATE=${BATCHES_PER_UPDATE:-1}
INCLUDE_BASELINES=${INCLUDE_BASELINES:-0}

if [ ! -f "$DATA" ]; then
    echo "[error] frozen snapshot not found: $DATA" >&2
    exit 1
fi
mkdir -p "$LOGS" "$OUT"

COMMON=(--protocol a --data-in "$DATA" --batch-size "$BATCH_SIZE"
    --batches-per-update "$BATCHES_PER_UPDATE" --updates "$UPDATES"
    --eval-interval "$EVAL_INTERVAL" --n-sigma 16 --compile default)
STAB=(--a-rho-target 0.99 --stab-mode power --lambda-stab 10.0
    --stab-iters 10 --stab-batch 256)

pids=()
run() {
    local name=$1
    shift
    echo "[launch] ${name}" >&2
    nohup .venv/bin/python experiment/koopman-world-model/train_wm_only.py "$@" \
        --out "$OUT/wm_${name}.json" --save-wm "$LOGS/${name}.pt" \
        > "$LOGS/${name}.log" 2>&1 &
    pids+=("$!")
}

run mlp_512 --wm-type mlp "${COMMON[@]}" --eval-modes latent --hidden 512 512 512
run mlp_1280 --wm-type mlp "${COMMON[@]}" --eval-modes latent --hidden 1280 1280 1280

STATE_COMMON=(--wm-type koopman "${COMMON[@]}" --eval-modes latent
    --koopman-state-in-latent --koopman-rank-domain 8 --koopman-rank-action-domain 4
    --lambda-cons 0.1 --cons-target-detach --lambda-lift-phys 0.0 --lambda-lift-phi 0.1
    "${STAB[@]}")

run state_small "${STATE_COMMON[@]}" --lambda-rec-true 0.0 \
    --dz 128 --rank 8 --koopman-enc-hidden 256 256 --koopman-dec-hidden 256 256
run state_small_rec "${STATE_COMMON[@]}" --lambda-rec-true 0.1 \
    --dz 128 --rank 8 --koopman-enc-hidden 256 256 --koopman-dec-hidden 256 256
run state_large_rec "${STATE_COMMON[@]}" --lambda-rec-true 0.1 \
    --dz 256 --rank 16 --koopman-enc-hidden 1024 1024 --koopman-dec-hidden 1024 1024

if [ "$INCLUDE_BASELINES" = "1" ]; then
    run mlp_2048 --wm-type mlp "${COMMON[@]}" --eval-modes latent --hidden 2048 2048 2048
    run koopman --wm-type koopman "${COMMON[@]}" --eval-modes latent,reencode \
        --dz 128 --rank 8 --koopman-enc-hidden 256 256 --koopman-dec-hidden 256 256 \
        "${STAB[@]}"
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
echo "[done] status=${status}; results in ${OUT}, logs in ${LOGS}" >&2
exit "$status"
