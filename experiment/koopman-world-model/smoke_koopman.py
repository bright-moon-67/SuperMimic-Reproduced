"""Smoke test for the Koopman world model path (temporary, mirrors test_world_model.py)."""

import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

NPZ = "data/npz/lafan1"
dev = "cuda:0"

files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=256, replay_capacity=400, seed=5), dev)
try:
    env.reset()
    for _ in range(300):
        env.step(0.3 * torch.randn(256, 29, device=env.device))
    rec = env.recorder_manager.get_term("replay")
    robot = env.scene["robot"]
    fk = G1Kinematics(load_g1_model(), device=env.device)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())

    cfg = WorldModelCfg(type="koopman", koopman_mode="latent")
    cfg.koopman.dz, cfg.koopman.rank = 64, 4
    cfg.koopman.enc_hidden = cfg.koopman.dec_hidden = (64, 64)
    wm = WorldModel(cfg, heads, env.device)
    lib = env.command_manager.get_term("motion").library
    tr = WorldModelTrainer(wm, lib, WMTrainerCfg(batch_size=512, batches_per_update=1, lr=1e-3, compile=None))
    e0 = tr.online_error(rec.buffer, n=1024, k=8)
    print("before:", [f"{v:.4f}" for v in e0.tolist()])
    for i in range(300):
        m = tr.update(rec.buffer)
        if i % 100 == 0:
            print(i, {k: f"{float(v):.4f}" for k, v in m.items()})
    e1 = tr.online_error(rec.buffer, n=1024, k=8)
    print("after: ", [f"{v:.4f}" for v in e1.tolist()])
    assert torch.isfinite(e1).all()
    assert e1[0] < 0.9 * e0[0] and e1[-1] < 0.9 * e0[-1], "koopman WM did not improve"

    # BPTT gradient through latent-propagation steps reaches the action input.
    w = rec.buffer.sample_windows(32, 4)
    a = w.a_exec.clone().requires_grad_(True)
    s = w.states[0]
    z = wm.encode_state(s, w.xi)
    for j in range(4):
        s, z = wm.step_z(z, s, a[j], w.xi)
    s.square().sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    print("BPTT grad OK, |grad|:", float(a.grad.abs().mean()))

    # reencode mode runs end-to-end too
    wm.cfg.koopman_mode = "reencode"
    m2 = tr.update(rec.buffer)
    print("reencode metrics:", {k: f"{float(v):.4f}" for k, v in m2.items()})
    wm.cfg.koopman_mode = "latent"
    print("SMOKE OK")
finally:
    env.close()
