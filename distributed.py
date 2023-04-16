import argparse
from pathlib import Path

from ablator.main.configs import SearchSpace
from omegaconf import OmegaConf
from ablator import ParallelTrainer, ParallelConfig, configclass, Results
import ray

from configs import PDModelConfig, PDTrainConfig
from utils import model_names
from wrapper import PDWrapper

# import os
# os.environ["RAY_ADDRESS"] = "127.0.0.1:4884"


@configclass
class PDParallelConfig(ParallelConfig):
    model_config: PDModelConfig
    train_config: PDTrainConfig


def my_train(config):
    yaml_kwargs = OmegaConf.load(Path(config).as_posix())
    kwargs = OmegaConf.to_object(yaml_kwargs)
    model = PDWrapper(
        model_class=model_names[kwargs["model_config"]["name"]]
    )

    run_config = PDParallelConfig(**kwargs)
    trainer = ParallelTrainer(
        wrapper=model,
        run_config=run_config,
    )
    trainer.gpu = 1 / run_config.concurrent_trials
    trainer.launch(kwargs['experiment_dir'], ray_head_address=None)

    res = Results(PDParallelConfig, trainer.experiment_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    kwargs = vars(args.parse_args())
    my_train(**kwargs)
