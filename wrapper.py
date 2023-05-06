import logging
from functools import cached_property
from typing import Callable, Dict, List, Optional

import numpy as np
import ablator
from tqdm import tqdm
from ablator import ModelWrapper
from ablator.main.configs import RunConfig

from configs import PDModelConfig, PDTrainConfig
from dataset import RLDataset
from models import BaseModel
from models.spd import SPD
from models.pd import PD
from modules.evaluation import PDMetrics
from utils import get_dataset, tensorboard_log_step

np.seterr(all="raise")

logger = logging.getLogger(__name__)


@ablator.configclass
class PDRunConfig(RunConfig):
    model_config: PDModelConfig
    train_config: PDTrainConfig


class PDWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs,
        )
        self.train_config: PDTrainConfig
        self.model_config: PDModelConfig
        self.model: BaseModel
        self.current_episode = 0

        self.train_dataloader: RLDataset
        self.test_dataloader: None

    def config_parser(self, config: RunConfig):
        config = super().config_parser(config)
        config.model_config.backbone_config = (
            self.train_dataloader.backbone_config
        )
        return config

    @cached_property
    def epoch_len(self):
        assert (
                hasattr(self, "train_dataloader") and len(self.train_dataloader) > 0
        ), "Undefined train_dataloader."
        return self.train_config.max_episodes

    @property
    def log_itr(self):
        return 1

    @property
    def current_epoch(self) -> int:
        return self.current_episode

    def evaluation_functions(self) -> Optional[Dict[str, Callable]]:
        return None

    def status_message(self) -> str:
        # must return current epoch, iter, losses and metrics
        msg = []
        for k, v in self.metrics.to_dict().items():
            if k in ['best_iteration', 'best_loss', 'current_epoch']:
                continue
            if 'loss' in k or 'reward' in k or 'buf_update' in k:
                msg.append(f"{k}: {v:.3f}")
        return f"Training episode [{self.current_episode}/{self.train_config.max_episodes}]: " \
               + " / ".join(msg)

    def train_loop(self, smoke_test=False):
        self.metrics = PDMetrics(
            tags=["train", "val"],
            batch_limit=float("inf"),
            val_episodes=self.run_config.train_config.val_episodes,
            static_aux_metrics=self.train_stats,
            moving_aux_metrics=["loss"] + getattr(self, "aux_metric_names", []),
        )

        self.train_online()
        return self.metrics

    def log(self):
        # Log step
        self.metrics.evaluate("train", reset=False)
        self.log_step()

    def train_online(self):
        # train model
        model = self.model
        max_episodes = self.train_config.max_episodes
        val_episodes = self.train_config.val_episodes

        print('Num student parameters:', sum(p.numel() for p in self.model.student.parameters() if p.requires_grad))

        for ep in range(max_episodes):
            self.current_episode = ep
            # train
            model.set_train()
            self.run_episode(tag="train", episode_id=ep)
            # eval
            if ep % 1 == 0:
                model.set_eval()
                for val_ep in range(val_episodes):
                    self.run_episode(tag="val", episode_id=val_ep)
                self.metrics.evaluate("val", reset=True)
                msg = self.status_message()
                self.logger.info(f"Evaluation Episode [{ep}] {msg}", verbose=False)

            self.update_status()
            self.log()

    def run_episode(self, tag, episode_id):
        model = self.model
        env = self.train_dataloader.get_dataloader()
        max_iter = self.train_dataloader.max_steps

        state = env.reset()
        for i in range(max_iter):
            # observe
            action = model.observe(state, tag)
            next_state, reward, terminate = env.step(action.numpy()[0])
            if tag == "train":
                model.add_data(states=state, actions=action, rewards=reward, next_states=next_state, masks=terminate)
                if model.online:
                    self.train_student(model)

            aux_metrics = {
                f'step_reward': reward,
                f'step_episode': np.array([episode_id]),
                f'buf_update': model.buffer_updates,
            }
            self.metrics.update_custom_metrics(aux_metrics, tag=tag)
            state = next_state.clone()
            if terminate:
                break

        # train
        if tag == "train" and not model.online:
            self.train_student(model)

    def train_student(self, model):
        train_iter = self.train_config.train_iter
        for _ in range(train_iter):
            batch = model.replay()
            _, train_metrics = self.train_step(batch)
            self.metrics.update_ma_metrics(train_metrics, tag="train")

    def make_dataloader_val(self, run_config: RunConfig):
        return None  # not a good idea to return wrong information i.e. train dataloader. Better to raise an error

    def make_dataloader_train(self, run_config: RunConfig):
        return get_dataset(run_config.train_config)
