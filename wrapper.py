import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import trainer
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm
from trainer import ModelWrapper
from trainer.config.run import RunConfigBase

from configs import ModelConfig, TrainConfig
from dataset import RLDataset
from models import BaseModel
from modules.evaluation import PDMetrics

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
        self.val_tasks: List[Dataset] = []

    def _total_steps(self):
        return self.train_config.max_episodes

    def config_parser(self, config: RunConfig):
        config = super().config_parser(config)
        config.model_config.backbone_config = (
            self.train_dataloader.make_backbone_config()
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

        for ep in range(max_episodes):
            # train
            model.student.train()
            model.teacher.reset()
            self.run_episode(tag="train", episode_id=ep)

            # eval
            if ep % 10 == 0:
                model.student.eval()
                self.metrics.get_preds("val").reset()
                for val_ep in range(10):
                    self.run_episode(tag="val", episode_id=val_ep)

                msg = self.metrics.get_msg()
                self.log_step()
                self.logger.info(f"Step [{ep}] {msg}", to_console=True)
                model.student.train()

            self.update_tqdm()

    def run_episode(self, tag, episode_id):
        model = self.model
        env = self.train_dataloader.get_dataloader()
        max_iter = 1000

        state = env.reset()
        for i in range(max_iter):
            # observe
            action = model.observe(state, tag)
            next_state, reward, terminate = env.step(action.numpy()[0])

            loss = 0.0
            if tag == "train":
                model.add_data(states=state, actions=action, rewards=reward, next_states=next_state, masks=terminate)
                batch = model.replay()
                _, _, loss, _ = self.train_step(batch)
                model.updates += 1

            aux_metrics = dict(rewards=reward[0], traj_names=str(episode_id))
            self.metrics.append(
                pred=None, labels=None, loss=loss, aux_metrics=aux_metrics, tag=tag,
            )

            state = next_state.clone()
            if terminate:
                break

        self.metrics.eval_metrics(tag)

    def make_dataloader_val(self, run_config: RunConfig):
        return None  # not a good idea to return wrong information i.e. train dataloader. Better to raise an error

    def make_dataloader_train(self, run_config: RunConfig):
        return RLDataset(run_config.train_config)
