import argparse
from pathlib import Path
from omegaconf import OmegaConf
from trainer import BaseTrainer, configclass, MPConfig
from trainer.mp import MPTrainer
import ray

from configs import ModelConfig, TrainConfig
from utils import model_names
from wrapper import PDWrapper

# import os
# os.environ["RAY_ADDRESS"] = "127.0.0.1:4884"


@configclass
class ParallelConfig(MPConfig):
    model_config: ModelConfig
    train_config: TrainConfig


def my_train(config):
    yaml_kwargs = OmegaConf.load(Path(config).as_posix())
    kwargs = OmegaConf.to_object(yaml_kwargs)
    model = PDWrapper(
        model_class=model_names[kwargs["model_config"]["name"]]
    )

    run_config = ParallelConfig(**kwargs)  # type: ignore
    run_config.train_config.tqdm = False
    ray.init(
        address="auto", runtime_env={"working_dir": Path(__file__).parent.resolve()}
    )
    trainer = MPTrainer(
        model=model,
        run_config=run_config,
        description="Parallel training experiments",
    )
    trainer.launch()

    from trainer.analysis.plot import CustomReport

    report = CustomReport(run_config)
    report.make()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    kwargs = vars(args.parse_args())
    my_train(**kwargs)
