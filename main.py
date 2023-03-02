import argparse
from pathlib import Path
from omegaconf import OmegaConf
from trainer import BaseTrainer

from utils import model_names
from wrapper import RunConfig, SCWrapper


def my_train(config):
    yaml_kwargs = OmegaConf.load(Path(config).as_posix())
    kwargs = OmegaConf.to_object(yaml_kwargs)
    run_config = RunConfig(**kwargs)
    model = SCWrapper(
        model_class=model_names[run_config.model_config.name]
    )

    trainer = BaseTrainer(
        model=model,
        run_config=run_config,
        description="Experiments",
    )

    trainer.launch()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    kwargs = vars(args.parse_args())
    my_train(**kwargs)
