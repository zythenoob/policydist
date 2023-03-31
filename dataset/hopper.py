import gymnasium as gym
import torch

from configs import BackboneConfig
from dataset import RLDataset


class Hopper(RLDataset):
    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make('Hopper-v4')
        self.pretrained = "edbeeching/decision-transformer-gym-hopper-medium"
        self.max_steps = 1000

    @property
    def state_mean(self):
        return torch.tensor([
            1.311279, -0.08469521, -0.5382719, -0.07201576, 0.04932366,
            2.1066856, -0.15017354, 0.00878345, -0.2848186, -0.18540096, -0.28461286,
        ])

    @property
    def state_std(self):
        return torch.tensor([
            0.17790751, 0.05444621, 0.21297139, 0.14530419, 0.6124444,
            0.85174465, 1.4515252, 0.6751696, 1.536239, 1.6160746, 5.6072536,
        ])

    @property
    def backbone_config(self):
        cfg = BackboneConfig(
            name="linear",
            input_dim=(11,),
            output_dim=3,
            dataset_name="hopper",
        )
        cfg.action_space = self.env.action_space
        return cfg
