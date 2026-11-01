"""Per-environment simulation context xi (plan: 33-d = foot friction 1 + encoder bias 29 + torso COM offset 3)."""

from __future__ import annotations

import numpy as np
import torch

XI_DIM = 33


class SimContext:
    def __init__(self, env, foot_geom_regex: str, com_body_name: str = "torso_link", entity_name: str = "robot"):
        self.env = env
        self.robot = env.scene[entity_name]
        local_ids, _names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )

    @torch.no_grad()
    def write(self, xi: torch.Tensor) -> None:
        """Restore friction / encoder bias / COM offset from a 33-d xi."""
        n = self.env.num_envs
        xi = xi.to(self.env.device, dtype=torch.float32)
        if xi.shape != (n, XI_DIM):
            raise ValueError(f"xi shape {tuple(xi.shape)} != {(n, XI_DIM)}")
        model = self.env.sim.model
        fr = model.geom_friction
        if fr.dim() == 3:
            fr[:, self.foot_geom_global, 0] = xi[:, 0]
        else:
            fr[self.foot_geom_global, 0] = xi[0, 0]
        self.robot.data.encoder_bias[:] = xi[:, 1:30]
        ipos = model.body_ipos
        if ipos.dim() == 3:
            ipos[:, self.com_body_global] = self.com_nominal + xi[:, 30:33]
        else:
            ipos[self.com_body_global] = self.com_nominal + xi[0, 30:33]
        self.env.sim.forward()

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        xi = torch.cat([friction.reshape(n, 1), bias, com], dim=-1)
        assert xi.shape == (n, XI_DIM)
        return xi.clone()
