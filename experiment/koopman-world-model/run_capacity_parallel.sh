#!/usr/bin/env bash
# P3 capacity sweep, run in parallel on one A800 (plan §6).
#
#   bash experiment/koopman-world-model/run_capacity_parallel.sh
#
# Four processes:
#   c1: Koopman dz128/r8, enc/dec 512      ~1.19M  (width-only)
#   c2: Koopman dz256/r16, enc/dec 512     ~1.62M
#   c3: Koopman dz256/r16, enc/dec 1024    ~4.50M
#   c4: MLP (420,420,420)                  ~0.44M  (matched baseline)
#
# Legacy diagnostic (not paper-aligned): the capacity sweep intentionally varies
# dz/rank/width, so it cannot match the upstream MLP baseline.  It keeps the
# original Koopman per-update budget (batch=2048, one batch/update, no compile).
# All share the frozen dataset and fixed A evaluation windows, seed=0, and the
# P2-B stability recipe (a_rho_target=0.99, sigma_max hinge, lambda_stab=10).
# Each process needs ~3-4 GB GPU memory (replay snapshot + env + model), so four
# concurrent runs fit comfortably on an 80 GB A800.  Check nvidia-smi; reduce
# concurrency if other jobs are running.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-/root/autodl-tmp/data/wm/lafan1.pt}
LOGS=${LOGS:-/root/autodl-tmp/logs/wm}
OUT=${OUT:-experiment/koopman-world-model}
mkdir -p "$LOGS"

COMMON=(--protocol a --data-in "$DATA" --a-rho-target 0.99 --stab-mode power --lambda-stab 10.0 --stab-iters 10 --stab-batch 256 --batch-size 2048 --batches-per-update 1 --compile none)

RUN_PID=""
run() {
    local name=$1
    shift
    echo "[launch] ${name}" >&2
    nohup .venv/bin/python experiment/koopman-world-model/train_wm_only.py "$@" \
        --out "$OUT/wm_${name}.json" \
        --save-wm "$LOGS/${name}.pt" \
        > "$LOGS/${name}.log" 2>&1 &
    RUN_PID=$!
}

pids=()
run c1 --wm-type koopman "${COMMON[@]}" --eval-modes latent,reencode --dz 128 --rank 8 --koopman-enc-hidden 512 512 --koopman-dec-hidden 512 512
pids+=("$RUN_PID")
run c2 --wm-type koopman "${COMMON[@]}" --eval-modes latent,reencode --dz 256 --rank 16 --koopman-enc-hidden 512 512 --koopman-dec-hidden 512 512
pids+=("$RUN_PID")
run c3 --wm-type koopman "${COMMON[@]}" --eval-modes latent,reencode --dz 256 --rank 16 --koopman-enc-hidden 1024 1024 --koopman-dec-hidden 1024 1024
pids+=("$RUN_PID")
run c4 --wm-type mlp --protocol a --data-in "$DATA" --eval-modes latent --hidden 420 420 420
pids+=("$RUN_PID")

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
echo "[done] status=${status}; logs in ${LOGS}" >&2
exit "$status"
