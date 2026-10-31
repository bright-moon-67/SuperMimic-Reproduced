#!/usr/bin/env bash
# Paper/upstream-aligned LAFAN1 world-model experiment.
#
#   bash experiment/koopman-world-model/run_paper_aligned.sh
#
# NOTE: this is the *offline* WM diagnostic (frozen replay snapshot).  For the
# exact online data distribution / WM-policy co-adaptation of the upstream
# repo, use run_lafan1_online.sh instead.
#
# Alignment target: origin/main `lafan1_a800` (the upstream LAFAN1 baseline):
#   data     : all 40 LAFAN1 clips, pretrained-policy rollouts, return sampling,
#              8192 envs, replay capacity 1000, action noise 0.1
#   WM (MLP): hidden (2048, 2048, 2048), window 8, batch 4096 x 10 batches/update,
#              lr 2e-4, grad clip 1.0, compile "default"
#   schedule : 320k env steps x 0.4 UTD = 128k WM updates
#
# Koopman has no upstream counterpart.  It keeps the original Koopman architecture
# (dz=128, rank=8, 256-wide enc/dec) and optionally the P2-B stability recipe.
# The MLP and Koopman runs use exactly the same frozen data and batch schedule.
#
# Useful overrides:
#   DATA=... POLICY=... NUM_ENVS=8192 COLLECT_STEPS=2000
#   BATCH_SIZE=4096 BATCHES_PER_UPDATE=10 WM_LR=2e-4
#   UPDATES=3000          # fast smoke/diagnostic, not the full upstream budget
#   UPDATES=19200         # match the local lafan1_a800 checkpoint (48k iters x 0.4 UTD)
#   UPDATES=128000        # full upstream schedule (320k iters x 0.4 UTD, default)
#   EVAL_INTERVAL=5000
#   STAB=0                # disable P2-B (pure original Koopman, no stability term)
#   XI_STATE=1            # augment latent state with xi: z~=[z, xi]
#   RECOLLECT=1           # rebuild the frozen snapshot even if DATA exists
#
# Note: the upstream code default is 4096 envs / WM batch 2048; the actual
# `lafan1_a800` reference checkpoint was trained with 8192 envs / batch 4096.
# This script follows the reference checkpoint.  Set NUM_ENVS=4096 and
# BATCH_SIZE=2048 for the code-default profile.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-/root/autodl-tmp/data/wm/lafan1_paper.pt}
POLICY=${POLICY:-/root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt}
LOGS=${LOGS:-/root/autodl-tmp/logs/wm}
OUT=${OUT:-experiment/koopman-world-model}

NUM_ENVS=${NUM_ENVS:-8192}
COLLECT_STEPS=${COLLECT_STEPS:-2000}
UPDATES=${UPDATES:-128000}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
BATCH_SIZE=${BATCH_SIZE:-4096}
BATCHES_PER_UPDATE=${BATCHES_PER_UPDATE:-10}
WM_LR=${WM_LR:-2e-4}
STAB=${STAB:-1}
XI_STATE=${XI_STATE:-0}
RECOLLECT=${RECOLLECT:-0}

mkdir -p "$LOGS"

if [ ! -f "$POLICY" ]; then
    echo "[error] policy checkpoint not found: $POLICY" >&2
    echo "        set POLICY=/path/to/ckpt_latest.pt (the upstream LAFAN1 actor)" >&2
    exit 1
fi

if [ ! -f "$DATA" ] || [ "$RECOLLECT" = "1" ]; then
    echo "[1/3] collecting paper-aligned frozen snapshot -> $DATA" >&2
    .venv/bin/python experiment/koopman-world-model/collect_wm_data.py \
        --paper \
        --policy-ckpt "$POLICY" \
        --num-envs "$NUM_ENVS" \
        --collect-steps "$COLLECT_STEPS" \
        --sampling-mode return \
        --action-std 0.1 \
        --out "$DATA"
else
    echo "[1/3] using existing snapshot $DATA (set RECOLLECT=1 to rebuild)" >&2
fi

# Upstream MLP world model.  This is the reference the Koopman run must match.
echo "[2/3] training upstream-aligned MLP (hidden 2048^3, batch 4096x10, lr 2e-4)" >&2
.venv/bin/python experiment/koopman-world-model/train_wm_only.py \
    --wm-type mlp \
    --protocol a \
    --data-in "$DATA" \
    --eval-modes latent \
    --hidden 2048 2048 2048 \
    --batch-size "$BATCH_SIZE" \
    --batches-per-update "$BATCHES_PER_UPDATE" \
    --wm-lr "$WM_LR" \
    --grad-clip 1.0 \
    --weight-decay 0.0 \
    --compile default \
    --updates "$UPDATES" \
    --eval-interval "$EVAL_INTERVAL" \
    --save-wm "$LOGS/paper_mlp.pt" \
    --out "$OUT/wm_paper_mlp.json"

# Koopman with the same data and optimizer budget.  Upstream has no Koopman
# architecture, so dz/rank/width stay at the original Koopman experiment values.
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

echo "[3/3] training Koopman (dz=128, rank=8, batch 4096x10, P2-B=${STAB})" >&2
.venv/bin/python experiment/koopman-world-model/train_wm_only.py \
    --wm-type koopman \
    --protocol a \
    --data-in "$DATA" \
    --eval-modes latent,reencode \
    --dz 128 \
    --rank 8 \
    --koopman-enc-hidden 256 256 \
    --koopman-dec-hidden 256 256 \
    "${XI_ARGS[@]}" \
    --batch-size "$BATCH_SIZE" \
    --batches-per-update "$BATCHES_PER_UPDATE" \
    --wm-lr "$WM_LR" \
    --grad-clip 1.0 \
    --weight-decay 0.0 \
    --compile default \
    --updates "$UPDATES" \
    --eval-interval "$EVAL_INTERVAL" \
    "${STAB_ARGS[@]}" \
    --save-wm "$LOGS/paper_koopman.pt" \
    --out "$OUT/wm_paper_koopman.json"

echo "[done] logs in $LOGS" >&2
