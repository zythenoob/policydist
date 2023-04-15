from typing import List, Optional, Tuple, Literal, Any
import ablator
from ablator import ModelConfig, TrainConfig, Annotation, Derived


@ablator.configclass
class BackboneConfig(ModelConfig):
    name: str
    input_dim: int
    output_dim: int
    # action_space: None = None
    pretrained: str
    # dataset_name: str


@ablator.configclass
class PDModelConfig(ModelConfig):
    name: str
    buffer_size: int = 1e+5
    replay_size: int = 200
    teacher_smooth_factor: float = 1.0
    teacher_std: float = 0.01

    # backbone
    backbone_config: Derived[BackboneConfig] = None

    # SPD hparams
    recent_replay_ratio: float = 0.5
    sup_decay: float = 0.8
    threshold: float = 10.0
    direction_threshold: float = -1.0


@ablator.configclass
class PDTrainConfig(TrainConfig):
    # data
    dataset: Literal["hopper", "walker", "halfcheetah"] = "hopper"
    env_seed: int = -1
    dataset_root_path: str = "./data"
    device: str = "cuda:0"
    # train
    max_episodes: int = 1000
    val_episodes: int = 100
    train_iter: int = 100
