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
    dataset_name: str


@trainer.configclass
class ModelConfig(ModelConfigBase):
    name: str
    buffer_size: int = 1e+5
    replay_size: int = 200
    teacher_smooth_factor: float = 1.0
    teacher_std: float = 0.01

    # backbone
    backbone_config: trainer.Annotated[Optional[BackboneConfig], trainer.Derived] = None

    # SPD hparams
    recent_replay_ratio: float = 0.5
    sup_decay: float = 0.8
    threshold: float = 10.0


@trainer.configclass
class TrainConfig(TrainConfigBase):
    # data
    dataset: Literal["hopper", "pong"] = "hopper"
    env_seed: int = -1
    dataset_root_path: str = "./data"
    device: str = "cuda:0"
    # train
    max_episodes: int = 1000
    val_episodes: int = 100
    train_iter: int = 100
