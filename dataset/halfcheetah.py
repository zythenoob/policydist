import gymnasium as gym
import torch

from configs import BackboneConfig
from dataset import RLDataset


class HalfCheetah(RLDataset):
    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make('HalfCheetah-v4')
        self.max_steps = 1000

    @property
    def state_mean(self):
        return torch.tensor([
            -0.04489148, 0.03232588, 0.06034835, -0.17081226, -0.19480659,
            -0.05751596, 0.09701628, 0.03239211, 11.047426, -0.07997331,
            -0.32363534, 0.36297753, 0.42322603, 0.40836546, 1.1085187,
            -0.4874403, -0.0737481
        ])

    @property
    def state_std(self):
        return torch.tensor([
            0.04002118, 0.4107858, 0.54217845, 0.41522816, 0.23796624,
            0.62036866, 0.30100912, 0.21737163, 2.2105937, 0.572586,
            1.7255033, 11.844218, 12.06324, 7.0495934, 13.499867,
            7.195647, 5.0264325
        ])

    @property
    def backbone_config(self):
        cfg = BackboneConfig(
            name="linear",
            input_dim=(17,),
            output_dim=6,
            pretrained="edbeeching/decision-transformer-gym-halfcheetah-expert",
        )
        cfg.action_space = self.env.action_space
        return cfg
