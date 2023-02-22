import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


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
        qnet_input_size = config.input_dim[0] + config.output_dim[0]
        self.Q1 = get_backbone("linear", qnet_input_size, 1)
        self.Q2 = get_backbone("linear", qnet_input_size, 1)

    def forward(self, state, action):
        qnet_input = torch.cat([state, action], dim=1)
        q1 = self.Q1(qnet_input)
        q2 = self.Q2(qnet_input)
        return q1, q2


class GaussianPolicy(nn.Module):
    def __init__(self, config):
        super(GaussianPolicy, self).__init__()
        action_space = config.action_space
        input_dim = config.input_dim[0]
        output_dim = config.output_dim[0]
        hidden_dim = 100

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, output_dim)
        self.log_std_linear = nn.Linear(hidden_dim, output_dim)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


def get_backbone(name, input_dim, output_dim):
    if name == "linear":
        return MLP(input_dim, output_dim)
    else:
        raise NotImplementedError
