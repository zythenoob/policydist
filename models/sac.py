import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import BaseModel
from modules.backbone import ValueNetwork, SoftQNetwork, PolicyNetwork
from modules.utils import hard_update, soft_update


class SAC(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.value_net = ValueNetwork(config.backbone_config)
        self.target_value_net = ValueNetwork(config.backbone_config)
        hard_update(self.target_value_net, self.value_net)

        self.soft_q_net = SoftQNetwork(config.backbone_config)
        self.policy_net = PolicyNetwork(config.backbone_config)

        self.gamma = config.discount_factor
        self.tau = config.soft_update_factor
        self.target_update_freq = config.target_update_freq

    @torch.no_grad()
    def observe(self, state):
        self.policy_net.eval()
        state = state.to(self.device)
        action = self.policy_net.sample_action(state)
        self.policy_net.train()
        return action.detach().cpu()

    def forward(self, states, actions, next_states, rewards, masks):
        states, actions, next_states, rewards, masks = (
            states.to(self.device),
            actions.to(self.device),
            next_states.to(self.device),
            rewards.unsqueeze(1).to(self.device),
            masks.unsqueeze(1).to(self.device),
        )

        expected_q_value = self.soft_q_net(states, actions)
        expected_value = self.value_net(states)
        new_action, log_prob, z, mean, log_std = self.policy_net(states)

        target_value = self.target_value_net(next_states)
        next_q_value = rewards + (1 - masks) * self.gamma * target_value
        q_value_loss = F.mse_loss(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(states, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = F.mse_loss(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        # regularization
        # mean_loss = mean_lambda * mean.pow(2).mean()
        # std_loss = std_lambda * log_std.pow(2).mean()
        # z_loss = z_lambda * z.pow(2).sum(1).mean()
        # policy_loss += mean_loss + std_loss + z_loss

        if self.updates % self.target_update_freq == 0:
            soft_update(self.target_value_net, self.value_net, self.tau)

        loss = q_value_loss + value_loss + policy_loss

        return None, loss
