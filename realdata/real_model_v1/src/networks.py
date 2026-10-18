"""Actor / critic / value networks shared across the algorithms."""
import torch
import torch.nn as nn
from . import config as C


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic tanh actor scaled to the twist bounds (TD3+BC)."""
    def __init__(self, obs_dim, act_dim, a_scale, hidden=C.HIDDEN):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim])
        self.register_buffer("a_scale", torch.as_tensor(a_scale, dtype=torch.float32))

    def forward(self, o):
        return torch.tanh(self.net(o)) * self.a_scale


class GaussianActor(nn.Module):
    """Squashed-Gaussian stochastic actor (IQL / CQL / PPO)."""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, act_dim, a_scale, hidden=C.HIDDEN, state_std=False):
        super().__init__()
        self.body = mlp([obs_dim, hidden, hidden], out_act=nn.ReLU)
        self.mu = nn.Linear(hidden, act_dim)
        self.state_std = state_std
        if state_std:
            self.log_std_layer = nn.Linear(hidden, act_dim)
        else:
            self.log_std_param = nn.Parameter(torch.zeros(act_dim))
        self.register_buffer("a_scale", torch.as_tensor(a_scale, dtype=torch.float32))

    def _mu_logstd(self, o):
        h = self.body(o)
        mu = self.mu(h)
        if self.state_std:
            log_std = self.log_std_layer(h)
        else:
            log_std = self.log_std_param.expand_as(mu)
        return mu, torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def forward(self, o, deterministic=False, with_logprob=True):
        mu, log_std = self._mu_logstd(o)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(std)
        a = torch.tanh(u)
        logp = None
        if with_logprob:
            logp = self._gauss_logp(u, mu, std) - self._tanh_corr(u)
        return a * self.a_scale, logp

    @staticmethod
    def _gauss_logp(u, mu, std):
        return (-0.5 * (((u - mu) / (std + 1e-6)) ** 2) - torch.log(std + 1e-6)
                - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)

    @staticmethod
    def _tanh_corr(u):
        return (2 * (torch.log(torch.tensor(2.0)) - u
                     - nn.functional.softplus(-2 * u))).sum(-1)

    def log_prob(self, o, a):
        mu, log_std = self._mu_logstd(o)
        std = log_std.exp()
        a_unit = torch.clamp(a / self.a_scale, -0.999999, 0.999999)
        u = torch.atanh(a_unit)
        return self._gauss_logp(u, mu, std) - self._tanh_corr(u)


class Critic(nn.Module):
    """Twin Q networks."""
    def __init__(self, obs_dim, act_dim, hidden=C.HIDDEN):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, o, a):
        return self.q1(torch.cat([o, a], -1)).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=C.HIDDEN):
        super().__init__()
        self.v = mlp([obs_dim, hidden, hidden, 1])

    def forward(self, o):
        return self.v(o).squeeze(-1)
