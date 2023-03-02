from typing import Optional

import torch
from gymnasium import Space

from configs import BackboneConfig
import gymnasium as gym


class RLDataset:
    def __init__(self, config):
        self.name = config.dataset
        self.type = config.dataset_type

        self.dataset = get_dataset(self.name)

    def step(self, action):
        assert self.type == "online"
        next_state, reward, terminate, *_ = self.dataset.step(action)
        return (
            torch.tensor(next_state).float().unsqueeze(0),
            torch.tensor(reward).float().unsqueeze(0),
            torch.tensor(terminate).bool().unsqueeze(0),
        )

    def reset(self):
        assert self.type == "online"
        state, _ = self.dataset.reset()
        return torch.tensor(state).float().unsqueeze(0)

    def get_dataloader(self):
        if self.type == "online":
            return self
        elif self.type == "offline":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def make_backbone_config(self):
        cfg = get_backbone_config(self.name, self.dataset.action_space)
        return cfg


def get_dataset(name: str):
    if name == "hopper":
        return gym.make('Hopper-v4')
    else:
        raise NotImplementedError


def get_backbone_config(name: str, action_space: Optional[Space] = None):
    if name == "hopper":
        cfg = BackboneConfig(
            name="linear",
            input_dim=(11,),
            output_dim=3,
        )
        cfg.action_space = action_space
        return cfg
