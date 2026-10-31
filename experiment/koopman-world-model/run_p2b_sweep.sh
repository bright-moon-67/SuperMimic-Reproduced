#!/usr/bin/env bash
# P2-B stability sweep on the frozen dataset (plan §5).
#
#   bash experiment/koopman-world-model/run_p2b_sweep.sh
#
# Legacy diagnostic (not paper-aligned): 3000 updates, window=8, batch=2048,
# one batch per update, no torch.compile.  The paper-aligned profile is
# run_paper_aligned.sh (batch=4096 x10, compile default).
# Each run: a_rho_target=0.99, sigma_max hinge with 10 power iterations on a
# 256-window subset.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-/root/autodl-tmp/data/wm/lafan1.pt}
OUT=${OUT:-experiment/koopman-world-model}
LOGS=${LOGS:-/root/autodl-tmp/logs/wm}
mkdir -p "$LOGS"

LAMBDAS=("$@")
if [ ${#LAMBDAS[@]} -eq 0 ]; then
    LAMBDAS=(0 1 10)
fi

for lam in "${LAMBDAS[@]}"; do
    name="p2b_l${lam}"
    echo "=== ${name} (lambda_stab=${lam}) ==="
    .venv/bin/python experiment/koopman-world-model/train_wm_only.py \
        --wm-type koopman --protocol a \
        --data-in "$DATA" \
        --a-rho-target 0.99 \
        --stab-mode power \
        --lambda-stab "$lam" \
        --stab-iters 10 \
        --stab-batch 256 \
        --batch-size 2048 \
        --batches-per-update 1 \
        --compile none \
        --eval-modes latent,reencode \
        --save-wm "$LOGS/${name}.pt" \
        --out "$OUT/wm_${name}.json"
done
