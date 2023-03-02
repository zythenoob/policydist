from typing import List, Optional, Tuple, Literal
import trainer
from gymnasium import Space
from trainer.config.run import ModelConfigBase, TrainConfigBase


@trainer.configclass
class BackboneConfig(ModelConfigBase):
    name: str
    input_dim: List[int]
    output_dim: int
    action_space = None


@trainer.configclass
class ModelConfig(ModelConfigBase):
    name: str
    buffer_size: int = 1e+5
    replay_size: int = 200
    # reward discount
    discount_factor: float = 0.99
    epsilon: float = 0.1
    # sac
    soft_update_factor: float = 0.005
    target_update_freq: int = 1000
    temperature: float = 0.2

    # backbone
    backbone_config: trainer.Annotated[Optional[BackboneConfig], trainer.Derived] = None
    pass


@trainer.configclass
class TrainConfig(TrainConfigBase):
    # data
    dataset: str = "mnist"
    dataset_type: str = "online"
    env_seed: int = -1
    dataset_root_path: str = "./data"
    device: str = "cuda:0"
    # train
    online_start_iter: int = 1000
    online_max_iters: int = 1e+6
    offline_epochs: int = 10
