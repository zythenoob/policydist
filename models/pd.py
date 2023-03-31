import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import BaseModel, DTModel
from modules.backbone import PolicyNetwork
from modules.memory import FIFOBuffer, PDBuffer, ReservoirBuffer
from modules.utils import mse_kd_loss, kl_div_kd_loss

"""
    Policy Distillation
"""
class PD(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.memory = FIFOBuffer(config.buffer_size)
        self.teacher = DTModel(config)
        self.student = PolicyNetwork(config.backbone_config)

        self.teacher_smooth_factor = config.teacher_smooth_factor
        self.teacher_std = config.teacher_std

    @torch.no_grad()
    def observe(self, state, tag):
        if tag == "train":
            return self.teacher.observe(state).cpu()
        else:
            # for computing loss
            self.teacher.observe(state)
            # use student action
            state = state.to(self.device)
            action = self.student.sample_action(state)
            return action.detach().cpu()

    def forward(self, states, actions, **kwargs):
        # https://github.com/CUN-bjy/policy-distillation-baselines/blob/main/classroom.py
        states, t_mean = states.to(self.device), actions.to(self.device)
        # get student action distribution
        s_dist = self.student(states)
        loss = self.compute_loss(s_dist, actions)
        return None, loss

    def compute_loss(self, s_dist, actions):
        t_mean = actions.to(self.device) / self.teacher_smooth_factor
        # deterministic
        s_mean, s_std = s_dist.loc, s_dist.scale
        deterministic_std = torch.ones(actions.shape[-1]) * self.teacher_std
        t_std = torch.stack([deterministic_std for _ in range(actions.shape[0])]).to(self.device)
        return kl_div_kd_loss([t_mean, t_std], [s_mean, s_std])

    def add_data(self, **kwargs):
        self.teacher.add_sequence_stats(kwargs['rewards'])
        self.memory.add_data(**kwargs)
