#!/usr/bin/env bash
# Quick offline comparison for the state-augmented Koopman:
#   z = [canonical(s); phi(s, xi)]
#   u = A z + B a + C xi + b
#       + sum_i a_i U_i V_i^T z
#       + sum_j xi_j P_j Q_j^T z
#       + sum_i a_i W_i xi
#   delta = W_d u[:71]; s_next = integrate(s, delta); z_next = [canonical(s_next); phi_next]
# No re-encoding inside the rollout.
#
#   bash experiment/koopman-world-model/run_state_koopman_quick.sh
#
# Uses the same 40-clip policy snapshot and 2000-update quick budget as the
# earlier quick runs.  Two capacities:
#   state_koopman       dz128/r8/256, domain r=8, action-domain r=4
#   state_koopman_large dz256/r16/1024, domain r=8, action-domain r=4
#
# Overrides:
#   DATA=... UPDATES=2000 EVAL_INTERVAL=500 BATCH_SIZE=4096
#   STAB=0       # disable P2-B
#   WITH_LARGE=0 # skip the 4.50M run
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-/root/autodl-tmp/data/wm/lafan1_paper_quick.pt}
LOGS=${LOGS:-/root/autodl-tmp/logs/wm}
OUT=${OUT:-experiment/koopman-world-model}
UPDATES=${UPDATES:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
BATCH_SIZE=${BATCH_SIZE:-4096}
BATCHES_PER_UPDATE=${BATCHES_PER_UPDATE:-1}
N_SIGMA=${N_SIGMA:-16}
LAMBDA_CONS=${LAMBDA_CONS:-0.1}
LAMBDA_LIFT_PHYS=${LAMBDA_LIFT_PHYS:-0.0}
LAMBDA_LIFT_PHI=${LAMBDA_LIFT_PHI:-0.1}
STAB=${STAB:-1}
WITH_LARGE=${WITH_LARGE:-1}

if [ ! -f "$DATA" ]; then
    echo "[error] frozen snapshot not found: $DATA" >&2
    echo "        run run_offline_quick_parallel.sh first (or set DATA=...)" >&2
    exit 1
fi
mkdir -p "$LOGS"

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
COMMON=(--protocol a --data-in "$DATA" --batch-size "$BATCH_SIZE" --batches-per-update "$BATCHES_PER_UPDATE" --updates "$UPDATES" --eval-interval "$EVAL_INTERVAL" --n-sigma "$N_SIGMA" --lambda-cons "$LAMBDA_CONS" --cons-target-detach --lambda-lift-phys "$LAMBDA_LIFT_PHYS" --lambda-lift-phi "$LAMBDA_LIFT_PHI" --compile default --koopman-state-in-latent --koopman-rank-domain 8 --koopman-rank-action-domain 4)

run state_koopman --wm-type koopman "${COMMON[@]}" --eval-modes latent,reencode \
    --dz 128 --rank 8 --koopman-enc-hidden 256 256 --koopman-dec-hidden 256 256 \
    "${STAB_ARGS[@]}"
pids+=("$RUN_PID")

if [ "$WITH_LARGE" = "1" ]; then
    run state_koopman_large --wm-type koopman "${COMMON[@]}" --eval-modes latent \
        --dz 256 --rank 16 --koopman-enc-hidden 1024 1024 --koopman-dec-hidden 1024 1024 \
        "${STAB_ARGS[@]}"
    pids+=("$RUN_PID")
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
echo "[done] status=${status}; logs in ${LOGS}" >&2
exit "$status"
