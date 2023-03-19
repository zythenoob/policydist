from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from transformers import DecisionTransformerModel

from modules.memory import FIFOBuffer, PrioritizedFIFOBuffer


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory = PrioritizedFIFOBuffer(config.buffer_size)
        self.replay_size = config.replay_size

        self.action_space = np.arange(config.backbone_config.output_dim)
        self.updates = 0
        pass

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def replay(self):
        return self.memory.get_data(self.replay_size)

    def add_data(self, **kwargs):
        self.memory.add_data(**kwargs)


class DTModel(BaseModel):
    SCALE = 1000.0  # normalization for rewards/returns
    TARGET_RETURN = 3600 / SCALE  # evaluation conditioning targets, 3600 is reasonable from the paper LINK

    def __init__(self, config):
        super().__init__(config)
        self.memory = None
        self.model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        self.model.eval()
        # sequence
        self.step = 0
        self.max_seq_len = self.model.config.max_length
        self.seq_attributes = ["target_returns", "states", "actions", "rewards", "timesteps"]
        self.reset()

    def reset(self):
        for attr_name in self.seq_attributes:
            setattr(self, attr_name, [])
        self.target_returns.append(torch.tensor(self.TARGET_RETURN).reshape(1, 1))
        self.timesteps.append(torch.tensor(self.step, dtype=torch.long).reshape(1, 1))

    @torch.no_grad()
    def observe(self, state):
        self.states.append(state)
        self.actions.append(torch.zeros((1, self.model.config.act_dim)))
        self.rewards.append(torch.zeros(1))
        state_seq = torch.cat(self.states, dim=0).to(self.device)
        action_seq = torch.cat(self.actions, dim=0).to(self.device)
        reward_seq = torch.cat(self.rewards, dim=0).to(self.device)
        target_return_seq = torch.cat(self.target_returns, dim=0).to(self.device)
        timestep_seq = torch.cat(self.timesteps, dim=0).to(self.device)
        # forward
        action = self.get_action(
            state_seq,
            action_seq,
            reward_seq,
            target_return_seq,
            timestep_seq,
        ).unsqueeze(0)
        self.actions[-1] = action.detach().cpu()
        return action.detach()

    def add_sequence_stats(self, reward):
        self.rewards[-1] = reward
        pred_return = self.target_returns[0] - (reward / self.SCALE)
        self.target_returns.append(pred_return)
        self.timesteps.append(torch.tensor(self.step, dtype=torch.long).reshape(1, 1))
        if len(self.states) > 2 * self.max_seq_len:
            for attr_name in self.seq_attributes:
                subsample_attr = getattr(self, attr_name)
                setattr(self, attr_name, subsample_attr[-self.max_seq_len:])

    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        # https://github.com/huggingface/transformers/blob/main/examples/research_projects/decision_transformer/run_decision_transformer.py
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.model.config.state_dim)
        actions = actions.reshape(1, -1, self.model.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.model.config.max_length is not None:
            states = states[:, -self.model.config.max_length:]
            actions = actions[:, -self.model.config.max_length:]
            returns_to_go = returns_to_go[:, -self.model.config.max_length:]
            timesteps = timesteps[:, -self.model.config.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(self.model.config.max_length - states.shape[1]), torch.ones(states.shape[1])]
            )
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (states.shape[0], self.model.config.max_length - states.shape[1], self.model.config.state_dim),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (actions.shape[0], self.model.config.max_length - actions.shape[1], self.model.config.act_dim),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (returns_to_go.shape[0], self.model.config.max_length - returns_to_go.shape[1], 1),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.model.config.max_length - timesteps.shape[1]), device=timesteps.device
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1]

