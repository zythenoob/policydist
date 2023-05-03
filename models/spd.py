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
        self.recent_replay_ratio = config.recent_replay_ratio
        self.sup_decay = config.sup_decay
        self.threshold = config.threshold
        self.direction_threshold = config.direction_threshold
        self.sample_surprise_count = config.sample_surprise_count
        self.to_sample = self.sample_surprise_count

        self.online = True

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
        self.update_surprise(loss)
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
        # sampling
        if not self.memory.is_full():
            self.memory.add_data(states=states, actions=actions, **kwargs)
            self.buffer_updates += 1
        else:
            # check surprise
            self.zero_grad()
            _, loss = self.forward(states, actions)
            loss.backward()
            surprise_score = self.check_surprise_magnitude()
            if surprise_score > self.threshold:
                surprise_direction = self.check_surprise_direction()
                if surprise_direction > self.direction_threshold:
                    self.to_sample = self.sample_surprise_count
            self.zero_grad()
            # on surprise
            if self.to_sample > 0:
                self.memory.add_data(states=states, actions=actions, **kwargs)
                self.to_sample -= 1
                self.buffer_updates += 1

    def replay(self, size=None):
        if size is None:
            size = self.replay_size
        recent_samples = int(size * self.recent_replay_ratio)
        return self.memory.get_data(size, recent=recent_samples)
    
    def check_surprise_magnitude(self):
        # check surprise
        grads = self.student.get_grads_list()
        grad_norm = torch.stack([torch.norm(g) for g in grads])
        surprise_score = z_score(grad_norm, **self.surprise_stats).abs().max()
        return surprise_score
    
    def check_surprise_direction(self):
        # check surprise
        grads = self.student.get_grads()
        buf_samples = self.memory.get_all_data()
        buf_length = len(self.memory)
        minibatch = 100
        minibatch_sim = []
        for idx in range(0, buf_length, minibatch):
            self.student.zero_grad()
            buf_states = buf_samples['states'][idx: idx + minibatch].to(self.device)
            buf_actions = buf_samples['actions'][idx: idx + minibatch].to(self.device)
            _, loss = self.forward(buf_states, buf_actions)
            loss.backward()
            buf_grads = self.student.get_grads()
            minibatch_sim.append((F.cosine_similarity(grads, buf_grads, dim=0) + 1).cpu().item() * len(buf_states))
            self.student.zero_grad()
        sim = np.sum(minibatch_sim) / buf_length
        return sim

    def update_surprise(self, loss):
        self.zero_grad()
        loss.backward(retain_graph=True)
        grads = self.student.get_grads_list()
        grad_norm = torch.stack([torch.norm(g) for g in grads])
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
        self.zero_grad()
