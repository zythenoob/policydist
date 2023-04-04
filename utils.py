import logging
import os
import importlib
from pathlib import Path

from trainer import SummaryLogger

import models
from dataset.halfcheetah import HalfCheetah
from dataset.hopper import Hopper
from dataset.walker import Walker
from modules.evaluation import PDMetrics


def get_all_models():
    return [
        model.stem
        if model.parent.name == "models"
        else model.parent.name + "." + model.stem
        for model in Path(models.__path__[0]).rglob("*.py")
        if not model.name.startswith("__")
    ]


model_names = {}
for model in get_all_models():
    import_comp = model.split(".")
    parent, model = "", import_comp[-1]
    if len(import_comp) > 1:
        parent = import_comp[0] + "."
    try:
        mod = importlib.import_module("models." + parent + model)
        class_name = {x.lower(): x for x in mod.__dir__()}[model.replace("_", "")]
        model_names[model] = getattr(mod, class_name)
    except:
        logging.warning(f"Could not load a baseline from: {model}")


def get_dataset(config):
    if config.dataset == "hopper":
        return Hopper(config)
    elif config.dataset == "walker":
        return Walker(config)
    elif config.dataset == "halfcheetah":
        return HalfCheetah(config)
    else:
        raise NotImplementedError

def tensorboard_log_step(
    writer: SummaryLogger, metrics: PDMetrics, model: models.BaseModel, iteration: int
):
    writer._add_metric(
        "updates", model.updates, iteration,
    )

