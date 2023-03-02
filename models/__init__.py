from abc import abstractmethod

import numpy as np
from torch import nn

from modules.memory import ReservoirBuffer, FIFOBuffer, PrioritizedFIFOBuffer


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory = PrioritizedFIFOBuffer(config.buffer_size)
        self.replay_size = config.replay_size

        self.action_space = np.arange(config.backbone_config.output_dim)
        self.updates = 0
        pass

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def replay(self):
        return self.memory.get_data(self.replay_size)
