import torch
import gymnasium as gym


class RLDataset:
    def __init__(self, config):
        self.name = config.dataset
        self.env: gym.Env

    @property
    def state_mean(self):
        raise NotImplementedError

    @property
    def state_std(self):
        raise NotImplementedError

    @property
    def backbone_config(self):
        raise NotImplementedError

    def step(self, action):
        next_state, reward, terminate, *_ = self.env.step(action)
        return (
            # [1, state_dim]
            ((torch.tensor(next_state).float() - self.state_mean) / self.state_std).unsqueeze(0),
            # [1, 1]
            torch.tensor(reward).float().unsqueeze(0),
            # [1, 1]
            torch.tensor(terminate).bool().unsqueeze(0),
        )

    def reset(self):
        state, _ = self.env.reset()
        return ((torch.tensor(state).float() - self.state_mean) / self.state_std).unsqueeze(0)

    def get_dataloader(self):
        return self
