"""Offline off-policy RL algorithms from scratch: TD3+BC, IQL, CQL.

Common interface:
    algo = Algo(obs_dim, act_dim, a_scale, obs_mean, obs_std, device)
    algo.update(batch)                  # (o, a, r, o2, term) normalised obs
    algo.act(obs_np, deterministic)     # numpy obs -> numpy action
"""
import copy
import torch
import torch.nn.functional as F

from . import config as C
from .networks import Actor, GaussianActor, Critic, ValueNet


class _Base:
    def __init__(self, obs_dim, act_dim, a_scale, obs_mean, obs_std, device):
        self.device = device
        self.act_dim = act_dim
        self.a_scale = torch.as_tensor(a_scale, dtype=torch.float32, device=device)
        self.obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32, device=device)
        self.obs_std = torch.as_tensor(obs_std, dtype=torch.float32, device=device)

    def _norm(self, o):
        return (o - self.obs_mean) / self.obs_std

    def act(self, obs_np, deterministic=True):
        o = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self._act_tensor(self._norm(o), deterministic)
        return a.squeeze(0).cpu().numpy()


class TD3BC(_Base):
    name = "TD3BC"

    def __init__(self, obs_dim, act_dim, a_scale, obs_mean, obs_std, device):
        super().__init__(obs_dim, act_dim, a_scale, obs_mean, obs_std, device)
        self.actor = Actor(obs_dim, act_dim, a_scale).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_t = copy.deepcopy(self.critic)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=C.LR_ACTOR)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=C.LR_CRITIC)
        self.it = 0

    def _act_tensor(self, o, deterministic=True):
        return self.actor(o)

    def update(self, b):
        self.it += 1
        o, a, r, o2, d = b
        with torch.no_grad():
            noise = (torch.randn_like(a) * C.TD3_POLICY_NOISE
                     ).clamp(-C.TD3_NOISE_CLIP, C.TD3_NOISE_CLIP) * self.a_scale
            a2 = (self.actor_t(o2) + noise).clamp(-self.a_scale, self.a_scale)
            q1t, q2t = self.critic_t(o2, a2)
            y = r + C.GAMMA * (1 - d) * torch.min(q1t, q2t)
        q1, q2 = self.critic(o, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        pil = 0.0
        if self.it % C.TD3_POLICY_FREQ == 0:
            pi = self.actor(o)
            q = self.critic.q1_only(o, pi)
            lam = C.TD3BC_ALPHA / (q.abs().mean().detach() + 1e-6)
            pi_loss = -lam * q.mean() + F.mse_loss(pi, a)
            self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.mul_(1 - C.TAU).add_(C.TAU * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.mul_(1 - C.TAU).add_(C.TAU * p.data)
            pil = float(pi_loss.item())
        return dict(q_loss=float(q_loss.item()), pi_loss=pil)

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict())

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"]); self.critic.load_state_dict(sd["critic"])


class IQL(_Base):
    name = "IQL"

    def __init__(self, obs_dim, act_dim, a_scale, obs_mean, obs_std, device):
        super().__init__(obs_dim, act_dim, a_scale, obs_mean, obs_std, device)
        self.actor = GaussianActor(obs_dim, act_dim, a_scale, state_std=True).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_t = copy.deepcopy(self.critic)
        self.value = ValueNet(obs_dim).to(device)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=C.LR_ACTOR)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=C.LR_CRITIC)
        self.v_opt = torch.optim.Adam(self.value.parameters(), lr=C.LR_CRITIC)

    def _act_tensor(self, o, deterministic=True):
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a

    @staticmethod
    def _expectile(diff, tau):
        return (torch.where(diff > 0, tau, 1 - tau) * diff ** 2).mean()

    def update(self, b):
        o, a, r, o2, d = b
        with torch.no_grad():
            q1, q2 = self.critic_t(o, a)
            q = torch.min(q1, q2)
        v = self.value(o)
        v_loss = self._expectile(q - v, C.IQL_EXPECTILE)
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        with torch.no_grad():
            y = r + C.GAMMA * (1 - d) * self.value(o2)
        q1, q2 = self.critic(o, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        with torch.no_grad():
            q1, q2 = self.critic_t(o, a)
            adv = torch.min(q1, q2) - self.value(o)
            w = torch.clamp(torch.exp(C.IQL_BETA * adv), max=C.IQL_MAX_WEIGHT)
        logp = self.actor.log_prob(o, a)
        pi_loss = -(w * logp).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.mul_(1 - C.TAU).add_(C.TAU * p.data)
        return dict(v_loss=float(v_loss.item()), q_loss=float(q_loss.item()),
                    pi_loss=float(pi_loss.item()))

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    value=self.value.state_dict())

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"]); self.critic.load_state_dict(sd["critic"])
        self.value.load_state_dict(sd["value"])


class CQL(_Base):
    name = "CQL"

    def __init__(self, obs_dim, act_dim, a_scale, obs_mean, obs_std, device):
        super().__init__(obs_dim, act_dim, a_scale, obs_mean, obs_std, device)
        self.actor = GaussianActor(obs_dim, act_dim, a_scale, state_std=True).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_t = copy.deepcopy(self.critic)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=C.LR_ACTOR)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=C.LR_CRITIC)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.ent_opt = torch.optim.Adam([self.log_alpha], lr=C.LR_ACTOR)
        self.target_entropy = -float(act_dim)
        self.n_act = C.CQL_N_ACTIONS

    def _act_tensor(self, o, deterministic=True):
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a

    def _lse(self, o):
        B = o.shape[0]
        rand_a = (torch.rand(B, self.n_act, self.act_dim, device=self.device) * 2 - 1) * self.a_scale
        o_rep = o.unsqueeze(1).expand(-1, self.n_act, -1).reshape(-1, o.shape[-1])
        q1r, q2r = self.critic(o_rep, rand_a.reshape(-1, self.act_dim))
        rand_logp = -torch.log(2 * self.a_scale).sum()
        pi_a, pi_lp = self.actor(o_rep, with_logprob=True)
        q1p, q2p = self.critic(o_rep, pi_a)

        def cat(qr, qp, lp):
            qr = qr.reshape(B, self.n_act) - rand_logp
            qp = qp.reshape(B, self.n_act) - lp.reshape(B, self.n_act).detach()
            return torch.logsumexp(torch.cat([qr, qp], 1), 1)
        return cat(q1r, q1p, pi_lp), cat(q2r, q2p, pi_lp)

    def update(self, b):
        o, a, r, o2, d = b
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self.actor(o2, with_logprob=True)
            q1t, q2t = self.critic_t(o2, a2)
            y = r + C.GAMMA * (1 - d) * (torch.min(q1t, q2t) - alpha * lp2)
        q1, q2 = self.critic(o, a)
        bellman = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        l1, l2 = self._lse(o)
        cons = (l1 - q1).mean() + (l2 - q2).mean()
        q_loss = bellman + C.CQL_ALPHA * cons
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        pi, lp = self.actor(o, with_logprob=True)
        q1p, q2p = self.critic(o, pi)
        pi_loss = (alpha * lp - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
        ent_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.ent_opt.zero_grad(); ent_loss.backward(); self.ent_opt.step()

        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.mul_(1 - C.TAU).add_(C.TAU * p.data)
        return dict(q_loss=float(bellman.item()), cons=float(cons.item()),
                    pi_loss=float(pi_loss.item()))

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict())

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"]); self.critic.load_state_dict(sd["critic"])


OFFLINE_ALGOS = {"TD3BC": TD3BC, "IQL": IQL, "CQL": CQL}
