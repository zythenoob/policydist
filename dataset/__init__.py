from configs import BackboneConfig
from dataset.hopper import Hopper
import gymnasium as gym


class RLDataset:
    def __init__(self, config):
        self.name = config.dataset
        self.type = config.dataset_type

        self.dataset = get_dataset(self.name)

    def get_dataloader(self):
        if self.type == "online":
            return self.dataset
        elif self.type == "offline":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def make_backbone_config(self):
        cfg = get_backbone_config(self.name)
        cfg.update = {"action_space": self.dataset.action_space}


def get_dataset(name: str):
    if name == "hopper":
        return gym.make('Hopper-v4')
    else:
        raise NotImplementedError

def get_backbone_config(name: str):
    if name == "hopper":
        return BackboneConfig(
            name="linear",
            input_dim=(11,),
            output_dim=(3,),
        )
