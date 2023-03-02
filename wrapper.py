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


class SCWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs,
        )
        self.train_config: TrainConfig
        self.model_config: ModelConfig
        self.model: BaseModel
        # self.metrics: SCMetrics

        self.train_dataloader: RLDataset
        self.test_dataloader: None
        self.val_tasks: List[Dataset] = []

    def _total_steps(self):
        return self.train_config.online_max_iters

    def config_parser(self, config: RunConfig):
        config = super().config_parser(config)
        config.model_config.backbone_config = (
            self.train_dataloader.make_backbone_config()
        )
        return config

    @property
    def epoch_len(self):
        return self.train_config.online_max_iters

    @property
    def log_itr(self):
        return 1

    def evaluation_functions(self) -> Optional[Dict[str, Callable]]:
        return None

    def train_loop(self):
        self.metrics = PDMetrics(
            epochs=self.train_config.online_max_iters,
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

        if self.train_dataloader.type == "online":
            self.train_online()
        elif self.train_dataloader.type == "offline":
            self.train_offline()
        else:
            raise NotImplementedError
        return self.metrics

    def log_step(self):
        super().log_step()

    def train_online(self):
        # train model
        model = self.model
        start_iter = self.train_config.online_start_iter
        max_iters = self.train_config.online_max_iters

        env = self.train_dataloader.get_dataloader()
        trajectory_id = 0
        model.train()

        state = env.reset()
        for iter in range(max_iters):
            # observe
            action = model.observe(state)
            next_state, reward, terminate = env.step(action.numpy()[0])

            model.memory.add_data(state, action, reward, next_state, terminate)

            loss = 0.0
            if iter > start_iter:
                batch = model.replay()
                _, _, loss, _ = self.train_step(batch)
                model.updates += 1

            aux_metrics = dict(rewards=reward[0], traj_names=str(trajectory_id))
            self.metrics.append_train(
                pred=None, labels=None, loss=loss, aux_metrics=aux_metrics
            )
            self.metrics.eval_metrics("train")

            state = next_state.clone()
            if terminate:
                state = env.reset()
                trajectory_id += 1
            self.update_tqdm()

            if iter % 500 == 0:
                msg = self.metrics.get_msg()
                self.log_step()
                self.logger.info(f"Step [{iter}] {msg}", to_console=False)

    def train_offline(self):
        raise NotImplementedError
        # train model
        model = self.model
        epochs = self.train_config.offline_epochs
        trajectory_loader = self.train_dataloader.get_dataloader()

        # for debugging
        print()
        print()
        print(self.train_dataloader.name, "Length:", len(trajectory_loader))

        # TODO Do not current_iteration because it affects logger. TQDM issue... Find a way around this
        # TODO 2: TQDM is messed up, why is that? has to do with epoch_len
        self.metrics.current_task_iteration = 0
        for e in range(epochs):
            model.train()
            for i, batch in enumerate(trajectory_loader):
                preds, labels, loss, _ = self.train_step(batch)

                # aux_metrics = dict(
                #     context_switch=model.surprise_state,
                #     score=model.surprise_score,
                #     ds_name=ds_name,
                #     warm_up=model.warmup,
                # )
                self.metrics.append_train(preds, labels, loss)
                self.update_tqdm()

            self.train_tqdm.reset()

    def make_dataloader_val(self, run_config: RunConfig):
        return None  # not a good idea to return wrong information i.e. train dataloader. Better to raise an error

    def make_dataloader_train(self, run_config: RunConfig):
        return RLDataset(run_config.train_config)
