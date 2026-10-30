"""Koopman full training-loop smoke: WM fitting + imagined BPTT + torch.compile (temporary)."""

import tempfile
from pathlib import Path

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer

tmp = Path(tempfile.mkdtemp(prefix="koopman_smoke_"))
cfg = TrainConfig(run_name="smoke_k", log_root=tmp, seed=1)
cfg.data.clips = ["walk1_subject1"]
cfg.env.num_envs = 128
cfg.env.replay_capacity = 200
cfg.world_model.type = "koopman"
cfg.world_model.koopman_mode = "latent"
cfg.world_model.koopman.dz = 64
cfg.world_model.koopman.rank = 4
cfg.world_model.koopman.enc_hidden = (64, 64)
cfg.world_model.koopman.dec_hidden = (64, 64)
cfg.policy.encoder_hidden = (256, 128)
cfg.policy.decoder_hidden = (256, 128)
cfg.wm_trainer.batch_size = 256
cfg.wm_trainer.batches_per_update = 2
cfg.wm_trainer.compile = "default"  # exercise the compiled _loss_step_z
cfg.policy_trainer.batch_size = 128
cfg.policy_trainer.horizon = 8
cfg.policy_trainer.compile = "default"  # exercise the compiled _step_z
cfg.schedule.total_iterations = 60
cfg.schedule.warmup_iterations = 20
cfg.schedule.log_interval = 20
cfg.schedule.checkpoint_interval = 60

tr = Trainer(cfg)
try:
    tr.run()
    assert tr.num_updates > 0, "no updates ran"
    ckpt = tmp / "smoke_k" / "ckpt_latest.pt"
    assert ckpt.exists(), "checkpoint missing"
    print("KOOPMAN TRAIN SMOKE OK, log at", tmp / "smoke_k")
finally:
    tr.env.close()
