"""PPO trained ONLINE in the data-grounded Go1GoalEnv.

PPO is on-policy, so it cannot consume the offline transitions directly.
Instead it learns inside the kinematic environment whose dynamics, speed limits
and map bounds are all taken from the real data.  Because episodes reset to
*random positions across the whole map*, PPO directly optimises the requested
objective: reach the goal from any start, respect the speed cap, stay in bounds.

A shared observation normaliser (running mean/std) is used; its stats are saved
with the policy so evaluation matches training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .networks import GaussianActor, ValueNet
from .env import Go1GoalEnv


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        bmean, bvar, bn = x.mean(0), x.var(0), x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bn
        self.mean += delta * bn / tot
        m_a = self.var * self.count
        m_b = bvar * bn
        self.var = (m_a + m_b + delta ** 2 * self.count * bn / tot) / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class PPO:
    name = "PPO"

    def __init__(self, obs_dim, act_dim, a_scale, device):
        self.device = device
        self.actor = GaussianActor(obs_dim, act_dim, a_scale, state_std=False).to(device)
        self.value = ValueNet(obs_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()), lr=C.PPO_LR)
        self.norm = RunningNorm(obs_dim)
        self.obs_dim, self.act_dim = obs_dim, act_dim

    # -- acting --------------------------------------------------------------
    def _pi(self, o_norm):
        a, logp = self.actor(o_norm, with_logprob=True)
        return a, logp

    def act(self, obs_np, deterministic=True):
        o = torch.as_tensor(self.norm.norm(obs_np), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    # -- training ------------------------------------------------------------
    def collect(self, env, n_steps):
        """Returns RAW observations (normalisation applied later at use-time)."""
        O, A, LP, R, D, V = [], [], [], [], [], []
        o = env.reset()
        for _ in range(n_steps):
            on = self.norm.norm(o)
            ot = torch.as_tensor(on, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a, logp = self.actor(ot, with_logprob=True)
                v = self.value(ot)
            a_np = a.squeeze(0).cpu().numpy()
            o2, r, done, info = env.step(a_np)
            O.append(o.copy()); A.append(a_np); LP.append(float(logp.item()))
            R.append(r); D.append(float(info["terminated"])); V.append(float(v.item()))
            o = env.reset() if done else o2
        with torch.no_grad():
            last_v = float(self.value(torch.as_tensor(self.norm.norm(o),
                          dtype=torch.float32, device=self.device).unsqueeze(0)).item())
        return (np.array(O, np.float32), np.array(A, np.float32),
                np.array(LP, np.float32), np.array(R, np.float32),
                np.array(D, np.float32), np.array(V, np.float32), last_v)

    def gae(self, R, D, V, last_v):
        n = len(R)
        adv = np.zeros(n, np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nextv = last_v if t == n - 1 else V[t + 1]
            nonterm = 1.0 - D[t]
            delta = R[t] + C.GAMMA * nextv * nonterm - V[t]
            gae = delta + C.GAMMA * C.PPO_GAE_LAMBDA * nonterm * gae
            adv[t] = gae
        ret = adv + V
        return adv, ret

    def train(self, total_steps=C.PPO_TOTAL_STEPS, log=print, eval_cb=None):
        env = Go1GoalEnv(rng=np.random.default_rng(C.SEED))
        history = []
        steps_done = 0
        it = 0
        while steps_done < total_steps:
            O, A, LP, R, D, V, last_v = self.collect(env, C.PPO_ROLLOUT)  # O is RAW
            self.norm.update(O)                       # update running stats on raw obs
            adv, ret = self.gae(R, D, V, last_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            Ot = torch.as_tensor(self.norm.norm(O).astype(np.float32), device=self.device)
            At = torch.as_tensor(A, device=self.device)
            LPt = torch.as_tensor(LP, device=self.device)
            ADt = torch.as_tensor(adv, device=self.device)
            RTt = torch.as_tensor(ret, device=self.device)

            idx = np.arange(len(O))
            for _ in range(C.PPO_EPOCHS):
                np.random.shuffle(idx)
                for s in range(0, len(O), C.PPO_MINIBATCH):
                    mb = idx[s:s + C.PPO_MINIBATCH]
                    mbi = torch.as_tensor(mb, device=self.device)
                    logp = self.actor.log_prob(Ot[mbi], At[mbi])
                    ratio = torch.exp(logp - LPt[mbi])
                    a1 = ratio * ADt[mbi]
                    a2 = torch.clamp(ratio, 1 - C.PPO_CLIP, 1 + C.PPO_CLIP) * ADt[mbi]
                    pi_loss = -torch.min(a1, a2).mean()
                    v_pred = self.value(Ot[mbi])
                    v_loss = F.mse_loss(v_pred, RTt[mbi])
                    # entropy proxy (encourage exploration)
                    _, logp_s = self.actor(Ot[mbi], with_logprob=True)
                    ent = -logp_s.mean()
                    loss = pi_loss + C.PPO_VF_COEF * v_loss - C.PPO_ENT_COEF * ent
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.value.parameters()),
                        C.PPO_MAX_GRAD_NORM)
                    self.opt.step()

            steps_done += C.PPO_ROLLOUT
            it += 1
            if eval_cb is not None and (it % 4 == 0 or steps_done >= total_steps):
                summ = eval_cb(self)
                summ["step"] = steps_done
                history.append(summ)
                log(f"[PPO] {steps_done:>8d} | succ {summ['success_rate']:.2f} "
                    f"| final_dist {summ['mean_final_dist']:.3f} "
                    f"| wall {summ['wall_hit_rate']:.2f} | steps {summ['mean_steps']:.0f}")
        return history

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), value=self.value.state_dict(),
                    norm_mean=self.norm.mean, norm_var=self.norm.var)

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        if "value" in sd:
            self.value.load_state_dict(sd["value"])
        self.norm.mean = np.asarray(sd["norm_mean"])
        self.norm.var = np.asarray(sd["norm_var"])
