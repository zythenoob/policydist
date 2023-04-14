import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import trainer
from tqdm import tqdm
from trainer import ModelWrapper
from trainer.config.run import RunConfigBase

from configs import ModelConfig, TrainConfig
from dataset import RLDataset
from models import BaseModel
from models.spd import SPD
from modules.evaluation import PDMetrics
from utils import get_dataset, tensorboard_log_step

np.seterr(all="raise")

logger = logging.getLogger(__name__)


@trainer.configclass
class RunConfig(RunConfigBase):
    model_config: ModelConfig
    train_config: TrainConfig


def collate_fn(p):
    return p


class PDWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs,
        )
        self.train_config: TrainConfig
        self.model_config: ModelConfig
        self.model: BaseModel

        self.train_dataloader: RLDataset
        self.test_dataloader: None

    def _total_steps(self):
        return self.train_config.max_episodes

    def config_parser(self, config: RunConfig):
        config = super().config_parser(config)
        config.model_config.backbone_config = (
            self.train_dataloader.backbone_config
        )
        return config

    @property
    def epoch_len(self):
        return self.train_config.max_episodes

    @property
    def log_itr(self):
        return 1

    def evaluation_functions(self) -> Optional[Dict[str, Callable]]:
        return None

    def train_loop(self):
        self.metrics = PDMetrics(
            epochs=self.train_config.max_episodes,
            total_steps=self._total_steps(),
            moving_average_limit=self.epoch_len,
            lr=self.run_config.train_config.optimizer_config.lr,
            evaluation_functions=self.evaluation_functions(),
            batch_limit=float("inf"),
            val_episodes=self.run_config.train_config.val_episodes,
        )

        self.train_tqdm = tqdm(
            total=self.epoch_len,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

        self.train_online()
        return self.metrics

    def log_step(self):
        super().log_step()

    def train_online(self):
        # train model
        model = self.model
        max_episodes = self.train_config.max_episodes
        val_episodes = self.train_config.val_episodes

        print('Num student parameters:', sum(p.numel() for p in self.model.student.parameters() if p.requires_grad))

        for ep in range(max_episodes):
            # train
            model.set_train()
            self.run_episode(tag="train", episode_id=ep)

            # eval
            if ep % 1 == 0:
                model.set_eval()
                self.metrics.get_preds("val").reset()
                for val_ep in range(val_episodes):
                    self.run_episode(tag="val", episode_id=val_ep)

                self.metrics.eval_metrics('val')
                msg = self.metrics.get_msg()
                self.logger.info(f"Step [{ep}] {msg}", to_console=True)

            self.metrics.eval_metrics('train')
            self.metrics.num_updates = model.updates
            self.update_tqdm()
            self.log_step()
            tensorboard_log_step(self.logger, self.metrics, model, self.iteration)

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
                if isinstance(model, SPD) and model.surprise_state:
                    self.train_student(model)
                    model.surprise_state = False

            aux_metrics = dict(rewards=reward[0], traj_names=str(episode_id))
            self.metrics.append(aux_metrics=aux_metrics, tag=tag)
            state = next_state.clone()
            if terminate:
                break

        # train
        if tag == "train":
            if not isinstance(model, SPD):
                self.train_student(model)

    def train_student(self, model):
        train_iter = self.train_config.train_iter
        for _ in range(train_iter):
            batch = model.replay()
            _, _, loss, _ = self.train_step(batch)
            self.metrics.append(loss=loss, tag="train")
            model.updates += 1

    def make_dataloader_val(self, run_config: RunConfig):
        return None  # not a good idea to return wrong information i.e. train dataloader. Better to raise an error

    def make_dataloader_train(self, run_config: RunConfig):
        return get_dataset(run_config.train_config)
