import gymnasium as gym
import torch

from configs import BackboneConfig
from dataset import RLDataset


class Walker(RLDataset):
    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make('Walker2d-v4')
        self.max_steps = 1000

    @property
    def state_mean(self):
        return torch.tensor([
            1.2384834e+00, 1.9578537e-01, -1.0475016e-01, -1.8579608e-01, 2.3003316e-01,
            2.2800924e-02, -3.7383768e-01, 3.3779100e-01, 3.9250960e+00, -4.7428459e-03,
            2.5267061e-02, -3.9287535e-03, -1.7367510e-02, -4.8212224e-01, 3.5432147e-04,
            -3.7124525e-03, 2.6285544e-03
        ])

    @property
    def state_std(self):
        return torch.tensor([
            0.06664903, 0.16980624, 0.17309439, 0.21843709, 0.74599105, 0.02410989,
            0.3729872, 0.6226182, 0.9708009, 0.72936815, 1.504065, 2.495893,
            3.511518, 5.3656907, 0.79503316, 4.317483, 6.1784487
        ])

    @property
    def backbone_config(self):
        cfg = BackboneConfig(
            name="linear",
            input_dim=17,
            output_dim=6,
            pretrained="edbeeching/decision-transformer-gym-walker2d-expert",
        )
        cfg.action_space = self.env.action_space
        return cfg
