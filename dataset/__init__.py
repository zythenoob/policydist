from typing import Optional

import torch
from gymnasium import Space

from configs import BackboneConfig
import gymnasium as gym


class RLDataset:
    def __init__(self, config):
        self.name = config.dataset

        self.dataset = get_dataset(self.name)
        self.state_mean, self.state_std = get_state_normalization(self.name)

    def step(self, action):
        next_state, reward, terminate, *_ = self.dataset.step(action)
        return (
            # [1, state_dim]
            ((torch.tensor(next_state).float() - self.state_mean) / self.state_std).unsqueeze(0),
            # [1, 1]
            torch.tensor(reward).float().unsqueeze(0),
            # [1, 1]
            torch.tensor(terminate).bool().unsqueeze(0),
        )

    def reset(self):
        state, _ = self.dataset.reset()
        return torch.tensor(state).float().unsqueeze(0)

    def get_dataloader(self):
        return self

    def make_backbone_config(self):
        cfg = get_backbone_config(self.name, self.dataset.action_space)
        return cfg


def get_dataset(name: str):
    if name == "hopper":
        return gym.make('Hopper-v4')
    else:
        raise NotImplementedError


def get_state_normalization(name: str):
    if name == "hopper":
        state_mean = torch.tensor([
            1.311279, -0.08469521, -0.5382719, -0.07201576, 0.04932366,
            2.1066856, -0.15017354, 0.00878345, -0.2848186, -0.18540096, -0.28461286,
        ])
        state_std = torch.tensor([
            0.17790751, 0.05444621, 0.21297139, 0.14530419, 0.6124444,
            0.85174465, 1.4515252, 0.6751696, 1.536239, 1.6160746, 5.6072536,
        ])
        return state_mean, state_std
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
