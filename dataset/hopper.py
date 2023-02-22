import gymnasium as gym

from configs import TrainConfig, BackboneConfig
from dataset import RLDataset


class Hopper(RLDataset):
    def __init__(self):
        super(Hopper, self).__init__()
        self.env = gym.make('Hopper-v4')

    def make_backbone_config(self):
        return BackboneConfig(
            name="resnet18",
            input_dim=SurpriseMNIST.DATA_SHAPE,
            output_dim=SurpriseMNIST.HEAD_SIZE,
        )
