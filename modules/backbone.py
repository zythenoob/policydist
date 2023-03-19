import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


# Initialize Policy weights
def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        hidden = 100
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (action space size)
        """
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.classifier(out)
        return out


class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = None
        qnet_input_size = config.input_dim[0] + config.output_dim
        self.Q1 = get_backbone("linear", qnet_input_size, 1)
        self.Q2 = get_backbone("linear", qnet_input_size, 1)

    def forward(self, state, action):
        qnet_input = torch.cat([state, action], dim=1)
        q1 = self.Q1(qnet_input)
        q2 = self.Q2(qnet_input)
        return q1, q2


class ValueNetwork(nn.Module):
    def __init__(self, config):
        super(ValueNetwork, self).__init__()
        input_dim = config.input_dim[0]
        self.net = get_backbone("linear", input_dim, 1)

    def forward(self, state):
        return self.net(state)


class SoftQNetwork(nn.Module):
    def __init__(self, config):
        super(SoftQNetwork, self).__init__()
        input_dim = config.input_dim[0] + config.output_dim
        self.net = get_backbone("linear", input_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.net(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, config, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        hidden_size = 100
        input_dim = config.input_dim[0]
        output_dim = config.output_dim

        self.net = get_backbone("linear", input_dim, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, output_dim)
        self.log_std_linear = nn.Linear(hidden_size, output_dim)
        self.apply(_weight_init)

    def forward(self, state):
        feats = self.net(state)

        mean = self.mean_linear(feats)
        log_std = self.log_std_linear(feats)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = log_std.exp()

        return Normal(mean, std)

    def sample_action(self, state):
        distribution = self.forward(state)
        return distribution.sample()


def get_backbone(name, input_dim, output_dim):
    if name == "linear":
        return MLP(input_dim, output_dim)
    else:
        raise NotImplementedError
