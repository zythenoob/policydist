from typing import Dict, Any

import numpy as np
from trainer.modules.main import Metrics, MovingAverage
from typing import List


class PDMetrics(Metrics):
    current_task_iteration: int = 0

    def __init__(self, val_episodes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.val_episodes = val_episodes
        self.num_updates = 0

    def eval_best(self, div_factor, div_warm_up_steps):
        is_best = False
        # Use val loss for scheduling or finding best checkpoint
        val_loss = self.val_loss
        is_best = val_loss < self.best_loss
        if is_best or self.best_loss == 0:
            self.best_iteration = self.current_iteration
            self.best_loss = val_loss

    @classmethod
    def _make_msg(cls, metrics: Dict[str, Any]):
        return " / ".join(
            [
                f"{k}: {v:.2e}"
                # else f"{k}: {v}"
                for k, v in metrics.items()
                if isinstance(v, (MovingAverage, np.number, int, float))
                and "aux" not in k
            ]
        )

    def get_msg(self):
        base_metrics = self.get_base_metrics()
        aux_metrics = self.get_added_metrics()
        base_msg = self._make_msg(base_metrics)
        aux_msg = self._make_msg(aux_metrics)

        msg = f"Training - " f"{base_msg} {aux_msg} "
        return msg

    def append(self, pred=None, labels=None, tag=None, loss=None, aux_metrics=None):
        super().append(pred, labels, tag, loss, aux_metrics)
        if aux_metrics is not None:
            limits = []
            for k, v in aux_metrics.items():
                tagged_arr = self.get_arr(f"{tag}_{k}")
                tagged_arr.append(v)
                limits.append(tagged_arr.limit)
            min_limit = min(limits)
            for k in aux_metrics.keys():
                tagged_arr = self.get_arr(f"{tag}_{k}")
                tagged_arr.limit = min_limit

    def eval_metrics(self, tag):
        super().eval_metrics(tag)
        aux_metrics = {}
        for k in self.__dict__.keys():
            if k.endswith("_arr__"):
                _tag, *name, _ = k.split("_")[2:-2]
                if _tag == tag:
                    name = "_".join(name)
                    aux_metrics[name] = self.get_arr(f"{tag}_{name}").arr

        self.update(self._make_aux_metrics(tag, **aux_metrics))

    def _make_aux_metrics(self, tag, rewards, traj_names, **kwargs):
        rewards = np.array(rewards)
        traj_switch_idx = np.nonzero(np.array(traj_names[1:]) != np.array(traj_names[:-1]))[0]
        cutoff = 10 if tag == "train" else self.val_episodes
        # avg reward of last 10 trajectories in training
        traj_switch_idx = traj_switch_idx[-cutoff:]
        avg_reward = 0
        if len(traj_switch_idx) > 0:
            avg_reward = np.sum(rewards[traj_switch_idx[0]:]) / len(traj_switch_idx)

        return {
            f'{tag}_reward': avg_reward,
        }

