import argparse
from pathlib import Path
from omegaconf import OmegaConf
from ablator import ProtoTrainer

from utils import model_names
from wrapper import PDWrapper, PDRunConfig


def my_train(config):
    yaml_kwargs = OmegaConf.load(Path(config).as_posix())
    kwargs = OmegaConf.to_object(yaml_kwargs)
    run_config = PDRunConfig(**kwargs)
    model = PDWrapper(
        model_class=model_names[run_config.model_config.name]
    )

    trainer = ProtoTrainer(
        wrapper=model,
        run_config=run_config,
    )

    trainer.launch()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    kwargs = vars(args.parse_args())
    my_train(**kwargs)
