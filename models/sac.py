import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import BaseModel
from modules.backbone import ValueNetwork, SoftQNetwork, PolicyNetwork


HPARAMS = {
    "gamma": 0.99,
    "tau": 0.01,
    "target_update_freq": 1,
}


def soft_update(target_net, net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(target_net, net):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(param.data)


class SAC(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.value_net = ValueNetwork(config.backbone_config)
        self.target_value_net = ValueNetwork(config.backbone_config)
        hard_update(self.target_value_net, self.value_net)

        self.soft_q_net = SoftQNetwork(config.backbone_config)
        self.policy_net = PolicyNetwork(config.backbone_config)

        self.gamma = HPARAMS['gamma']
        self.tau = HPARAMS['tau']
        self.target_update_freq = HPARAMS['target_update_freq']

    @torch.no_grad()
    def observe(self, state, tag):
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
        dist = self.policy_net(states)
        r = dist.rsample()
        z = dist.sample()
        new_action = torch.tanh(z)
        log_prob = (dist.log_prob(r) - torch.log(1 - new_action.pow(2) + 1e-8)).sum(1, keepdim=True)

        target_value = self.target_value_net(next_states)
        next_q_value = rewards + (1 - masks) * self.gamma * target_value
        q_value_loss = F.mse_loss(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(states, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = F.mse_loss(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        if self.updates % self.target_update_freq == 0:
            soft_update(self.target_value_net, self.value_net, self.tau)

        loss = q_value_loss + value_loss + policy_loss

        return None, loss
