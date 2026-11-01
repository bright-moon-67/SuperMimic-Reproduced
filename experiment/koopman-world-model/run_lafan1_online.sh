#!/usr/bin/env bash
# Online LAFAN1 alignment: the upstream `sm-train` loop (Algorithm A.1).
#
#   bash experiment/koopman-world-model/run_lafan1_online.sh
#
# Why this script exists: `run_paper_aligned.sh` trains an *offline* world model
# on a frozen replay snapshot.  The upstream repo trains the WM online inside the
# policy loop, so the data distribution, sampler state, update-to-data ratio and
# WM/policy co-adaptation are different.  This script is the aligned online path.
#
# Alignment target: origin/main `lafan1_a800` on LAFAN1 only:
#   data     : all 40 LAFAN1 clips, no BONES-SEED
#   env      : 8192 envs, 10 s episodes, action noise 0.1, obs/reset noise,
#              domain randomisation, return sampling, replay capacity 1000
#   WM       : MLP baseline is 2048^3; here type=koopman, dz=128, rank=8,
#              enc/dec 256, window 8, batch 4096 x 10, lr 2e-4, compile default
#   policy   : H=32, batch 8192, history 10, gamma 0.99, GAE 0.95, critic 5x4
#   schedule : 320k env steps, UTD 0.4 -> 128k WM updates, cosine decay
#
# Overrides:
#   NUM_ENVS=4096              # upstream code default (lafan1_a800 uses 8192)
#   TOTAL_ITERATIONS=48000     # match the local lafan1_a800 checkpoint length
#   MAX_HOURS=3                # upstream README's LAFAN1 run cap
#   STAB=0                     # disable P2-B (pure original Koopman)
#   XI_STATE=1                 # augment latent state with xi: z~=[z, xi]
#   STATE_KOOPMAN=1            # z=[canonical(s); phi(s,xi)], delta+integrate
#   RESUME=/path/ckpt.pt       # checkpoint to resume
#   RESUME_MODE=transfer_policy # load policy/critic only, fresh WM
#   RESUME_MODE=full            # restore weights + optimizers + iteration
#   RESUME_MODE=wm_only         # restore WM only
#   RESUME_SKIP=wm              # optional explicit skip (experiment B)
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-/root/autodl-tmp/data/npz/lafan1}
RUN_NAME=${RUN_NAME:-lafan1_koopman_aligned}
WM_TYPE=${WM_TYPE:-koopman}
MLP_HIDDEN=${MLP_HIDDEN:-"2048 2048 2048"}
LOG_ROOT=${LOG_ROOT:-/root/autodl-tmp/logs}
NUM_ENVS=${NUM_ENVS:-8192}
TOTAL_ITERATIONS=${TOTAL_ITERATIONS:-320000}
MAX_HOURS=${MAX_HOURS:-}
STAB=${STAB:-1}
XI_STATE=${XI_STATE:-0}
STATE_KOOPMAN=${STATE_KOOPMAN:-0}
RESUME=${RESUME:-}
RESUME_MODE=${RESUME_MODE:-full}
RESUME_SKIP=${RESUME_SKIP:-}

STAB_ARGS=()
if [ "$STAB" = "1" ]; then
    STAB_ARGS=(
        --world-model.koopman.a-rho-target 0.99
        --world-model.koopman.stab-mode power
        --world-model.koopman.lambda-stab 10.0
        --world-model.koopman.stab-iters 10
        --world-model.koopman.stab-batch 256
    )
fi
XI_ARGS=()
if [ "$XI_STATE" = "1" ]; then
    XI_ARGS=(--world-model.koopman.use-xi-state)
fi
STATE_ARGS=()
if [ "$STATE_KOOPMAN" = "1" ]; then
    STATE_ARGS=(
        --world-model.koopman.state-in-latent
        --world-model.koopman.rank-domain 8
        --world-model.koopman.rank-action-domain 4
        --world-model.koopman.lambda-cons "${LAMBDA_CONS:-0.1}"
        --world-model.koopman.cons-target-detach
        --world-model.koopman.lambda-lift-phys "${LAMBDA_LIFT_PHYS:-0.0}"
        --world-model.koopman.lambda-lift-phi "${LAMBDA_LIFT_PHI:-0.1}"
    )
fi

if [ "$WM_TYPE" = "mlp" ]; then
    WM_ARGS=(
        --world-model.type mlp
        --world-model.hidden $MLP_HIDDEN
    )
else
    WM_ARGS=(
        --world-model.type koopman
        --world-model.koopman-mode latent
        --world-model.koopman.dz 128
        --world-model.koopman.rank 8
        --world-model.koopman.enc-hidden 256 256
        --world-model.koopman.dec-hidden 256 256
    )
fi

MAX_HOURS_ARGS=()
if [ -n "$MAX_HOURS" ]; then
    MAX_HOURS_ARGS=(--schedule.max-hours "$MAX_HOURS")
fi

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
    RESUME_ARGS=(--resume "$RESUME" --resume-mode "$RESUME_MODE")
    if [ -n "$RESUME_SKIP" ]; then
        RESUME_ARGS+=(--resume-skip "$RESUME_SKIP")
    fi
fi

.venv/bin/sm-train \
    --run-name "$RUN_NAME" \
    --log-root "$LOG_ROOT" \
    --seed 0 \
    --device cuda:0 \
    --data.npz-dir "$DATA" \
    --env.num-envs "$NUM_ENVS" \
    --env.episode-length-s 10.0 \
    --env.action-noise-std 0.1 \
    --env.obs-noise \
    --env.reset-noise \
    --env.domain-randomization \
    --env.sampling-mode return \
    --env.replay-capacity 1000 \
    --env.future-frames 10 \
    --env.no-compact-library \
    --env.no-anchor-obs \
    "${WM_ARGS[@]}" "${STAB_ARGS[@]}" "${XI_ARGS[@]}" "${STATE_ARGS[@]}" \
    --wm-trainer.window 8 \
    --wm-trainer.batches-per-update 10 \
    --wm-trainer.batch-size 4096 \
    --wm-trainer.lr 2e-4 \
    --wm-trainer.grad-clip 1.0 \
    --wm-trainer.weight-decay 0.0 \
    --wm-trainer.compile default \
    --policy.encoder-hidden 2048 1024 512 256 \
    --policy.decoder-hidden 2048 1024 512 256 \
    --policy.latent-dim 32 \
    --policy.history 10 \
    --policy.goal-term-scale 1.0 5.0 1.0 2.0 1.0 1.0 \
    --policy.decoder-last-gain 0.01 \
    --critic.hidden 256 128 \
    --policy-trainer.horizon 32 \
    --policy-trainer.batch-size 8192 \
    --policy-trainer.history 10 \
    --policy-trainer.future-frames 10 \
    --policy-trainer.gamma 0.99 \
    --policy-trainer.gae-lambda 0.95 \
    --policy-trainer.lr 2e-4 \
    --policy-trainer.betas 0.9 0.95 \
    --policy-trainer.grad-clip-norm 1.0 \
    --policy-trainer.grad-clip-value 0.1 \
    --policy-trainer.critic-lr 2e-4 \
    --policy-trainer.critic-grad-clip 1.0 \
    --policy-trainer.critic-epochs 5 \
    --policy-trainer.critic-minibatches 4 \
    --policy-trainer.use-critic \
    --policy-trainer.obs-noise \
    --policy-trainer.compile default \
    --policy-trainer.no-anchor-obs \
    --schedule.total-iterations "$TOTAL_ITERATIONS" \
    --schedule.updates-per-env-step 0.4 \
    --schedule.policy-updates-per-wm-update 1 \
    --schedule.warmup-iterations 40 \
    --schedule.lr-final-ratio 0.03 \
    --schedule.lr-decay-fraction 0.78125 \
    --schedule.log-interval 50 \
    --schedule.checkpoint-interval 4000 \
    --schedule.keep-checkpoints 4 \
    "${MAX_HOURS_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
