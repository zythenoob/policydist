import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import BaseModel
from modules.backbone import GaussianPolicy, QNetwork
from modules.utls import hard_update


class SAC(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.policy = GaussianPolicy(config)
        self.critic = QNetwork(config)
        self.critic_target = QNetwork(config)
        hard_update(self.critic_target, self.critic)

        self.gamma = config.discount_factor
        self.epsilon = config.epsilon
        self.target_update_freq = config.target_update_freq

        self.loss = nn.SmoothL1Loss()

    def observe(self, state):
        if self.training:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def forward(self, states, actions, next_states, rewards, masks):
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + masks * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        pi, log_pi, _ = self.policy.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        loss = qf_loss + policy_loss

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return None, loss



