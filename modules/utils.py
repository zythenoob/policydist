import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F


def kl_div_kd_loss(teacher_dist_info, student_dist_info):
    # https://github.com/Mee321/policy-distillation/blob/master/utils2/math.py#L17
    pi = Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])
    pi_new = Normal(loc=student_dist_info[0], scale=student_dist_info[1])
    kl = torch.mean(kl_divergence(pi, pi_new))
    return kl

def mse_kd_loss(source, target):
    return F.mse_loss(source, target)
