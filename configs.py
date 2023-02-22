from typing import List, Optional, Tuple, Literal
import trainer
from gymnasium import Space
from trainer.config.run import ModelConfigBase, TrainConfigBase


@trainer.configclass
class BackboneConfig(ModelConfigBase):
    name: str
    input_dim: List[int]
    output_dim: int
    action_space: Space


@trainer.configclass
class ModelConfig(ModelConfigBase):
    name: str
    replay_size: int = 200
    # reward discount
    discount_factor: float = 0.99

    # backbone
    backbone_config: trainer.Annotated[Optional[BackboneConfig], trainer.Derived] = None
    pass


@trainer.configclass
class TrainConfig(TrainConfigBase):
    # data
    dataset: str = "mnist"
    env_seed: int = -1
    dataset_root_path: str = "./data"
    device: str = "cuda:0"
