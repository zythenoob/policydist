from typing import Dict, Any

import numpy as np
from typing import List

from ablator.modules.metrics.main import TrainMetrics
from ablator.modules.metrics.stores import ArrayStore
import ablator.utils.base as butils


class PDMetrics(TrainMetrics):
    current_task_iteration: int = 0

    def __init__(self, val_episodes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.val_episodes = val_episodes
        self.custom_attrs = ["step_reward", "step_episode"]
        self.custom_metrics = {}
        self.val_rewards = []
        self._init_custom_metrics()

    def to_dict(self):
        attrs = ['train_loss']
        ma_attrs = {k: self._get_ma(k).value for k in attrs}
        static_attrs = {k: getattr(self, k) for k in self.__static_aux_attributes__}
        return {**ma_attrs, **static_attrs, **self.custom_metrics}

    def update_custom_metrics(self, metric_dict: dict[str, Any], tag: str):
        metric_dict = butils.iter_to_numpy(metric_dict)
        for k, v in metric_dict.items():
            attr_name = f"{tag}_{k}"
            _arr = getattr(self, attr_name, None)
            assert _arr is not None, f"invalid custom metric {attr_name}"
            _arr.append(v)

    def _init_custom_metrics(self):
        for tag in self.__tags__:
            for attr in self.custom_attrs:
                attr_name = f"{tag}_{attr}"
                _arr = ArrayStore(
                    batch_limit=self.__batch_limit__,
                    memory_limit=self.__memory_limit__,
                )
                setattr(self, attr_name, _arr)

    def evaluate(self, tag, reset=True, update_ma=True):
        if tag == "val":
            self.custom_metrics.update(self.eval_rewards())
            for attr in self.custom_attrs:
                _arr = getattr(self, f"val_{attr}", None)
                _arr.reset()

    def eval_rewards(self):
        train_reward = self.get_avg_reward("train")
        val_reward = self.get_avg_reward("val")
        self.val_rewards.append(val_reward)
        avg_val_reward = np.mean(np.array(self.val_rewards)[-5:])

        return {
            f'train_reward': train_reward,
            f'val_reward': val_reward,
            f'avg_val_reward': avg_val_reward,
        }

    def get_avg_reward(self, tag):
        cutoff = 10 if tag == "train" else self.val_episodes
        rewards = getattr(self, f"{tag}_step_reward").get()[0]
        episodes = getattr(self, f"{tag}_step_episode").get()[0]
        ep_ids = np.unique(episodes)[-cutoff:]
        avg_reward = 0
        for ep in ep_ids:
            avg_reward += np.sum(rewards[episodes == ep])
        avg_reward /= len(ep_ids)
        return avg_reward
