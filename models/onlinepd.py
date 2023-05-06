import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.pd import PD
from modules.backbone import PolicyNetwork
from modules.memory import FIFOBuffer, PDBuffer, ReservoirBuffer
from modules.utils import mse_kd_loss, kl_div_kd_loss

"""
    Policy Distillation + Online
"""
class OnlinePD(PD):
    def __init__(self, config):
        super().__init__(config)
        self.memory = FIFOBuffer(config.buffer_size)
        self.online = True
