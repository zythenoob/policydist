import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import BaseModel, DTModel
from modules.backbone import PolicyNetwork
from modules.memory import FIFOBuffer
from modules.utils import mse_kd_loss, kl_div_kd_loss, z_score

"""
    Surprise Policy Distillation
"""
class SPD(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.memory = FIFOBuffer(config.buffer_size)
        self.teacher = DTModel(config)
        self.student = PolicyNetwork(config.backbone_config)

        self.teacher_smooth_factor = config.teacher_smooth_factor
        self.teacher_std = config.teacher_std

        self.surprise_state = False
        self.surprise_stats = {
            "mu": None,
            "std": None,
        }
        self.recent_replay_ratio = 0.5
        self.sup_decay = 0.8
        self.threshold = 10.0

    @torch.no_grad()
    def observe(self, state, tag):
        if tag == "train":
            return self.teacher.observe(state).cpu()
        else:
            # use student action
            state = state.to(self.device)
            action = self.student.sample_action(state)
            return action.detach().cpu()

    def forward(self, states, actions, **kwargs):
        # https://github.com/CUN-bjy/policy-distillation-baselines/blob/main/classroom.py
        states = states.to(self.device)
        # get student action distribution
        s_dist = self.student(states)
        loss = self.compute_loss(s_dist, actions)
        return None, loss

    def compute_loss(self, s_dist, actions):
        t_mean = actions.to(self.device) / self.teacher_smooth_factor
        s_mean, s_std = s_dist.loc, s_dist.scale
        # fake teacher std
        t_std = torch.ones(actions.shape[-1]) * self.teacher_std
        t_std = torch.stack([t_std for _ in range(actions.shape[0])]).to(self.device)
        return kl_div_kd_loss([t_mean, t_std], [s_mean, s_std])

    def add_data(self, states, actions, **kwargs):
        self.teacher.add_sequence_stats(kwargs['rewards'])
        # check surprise
        self.student.zero_grad()
        _, loss = self.forward(states, actions)
        loss.backward()
        grads = self.student.get_grads_list()
        grad_norm = torch.stack([torch.norm(g) for g in grads])
        surprise_score = z_score(grad_norm, **self.surprise_stats).abs().max()
        # update
        if self.surprise_stats["mu"] is None:
            self.surprise_stats["mu"] = torch.zeros_like(grad_norm)
            self.surprise_stats["std"] = torch.ones_like(grad_norm)
        delta = grad_norm - self.surprise_stats["mu"]
        delta = delta.clone()
        self.surprise_stats["mu"] += self.sup_decay * delta
        self.surprise_stats["std"] = (1 - self.sup_decay) * (
                self.surprise_stats["std"] + self.sup_decay * delta ** 2
        )
        self.student.zero_grad()

        if surprise_score > self.threshold:
            self.surprise_state = True
            self.memory.add_data(states=states, actions=actions, **kwargs)

    def replay(self):
        recent_samples = int(self.replay_size * self.recent_replay_ratio)
        return self.memory.get_data(self.replay_size, recent=recent_samples)
