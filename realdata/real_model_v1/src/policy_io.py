"""Load a saved policy checkpoint into a callable act(obs)->action."""
import numpy as np
import torch

from . import config as C
from .offline_algos import OFFLINE_ALGOS
from .ppo import PPO


def load_policy(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    name = ckpt["algo"]
    dev = torch.device(device)
    if name == "PPO":
        agent = PPO(C.OBS_DIM, C.ACT_DIM, ckpt["a_scale"], dev)
        sd = dict(ckpt["state_dict"])
        sd["norm_mean"] = np.asarray(ckpt["norm_mean"])
        sd["norm_var"] = np.asarray(ckpt["norm_var"])
        agent.load_state_dict(sd)
    else:
        agent = OFFLINE_ALGOS[name](C.OBS_DIM, C.ACT_DIM, ckpt["a_scale"],
                                    ckpt["obs_mean"], ckpt["obs_std"], dev)
        agent.load_state_dict(ckpt["state_dict"])
    meta = dict(name=name, a_scale=np.asarray(ckpt["a_scale"]),
                goal=ckpt["goal"], goal_radius=ckpt["goal_radius"],
                map=ckpt["map"], best=ckpt.get("best"), history=ckpt.get("history"))
    return (lambda o: agent.act(o, True)), meta
