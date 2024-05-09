"""
Trying raylib for the experiments
Likely mostly to serve as a benchmark of an industrial solution
"""
from argparse import ArgumentParser

import gymnasium as gym
import torch
from ray.rllib.algorithms import ppo

from sampleddetection.environment.models import Environment
from sampleddetection.samplers.window_sampler import DynamicWindowSampler


def argsies():
    ap = ArgumentParser()
    ap.add_argument(
        "--num_obs_elements",
        help="Number of observation elements. Likely amount of statistics to use to make a decision",
    )

    return ap.parse_args()


class MetaEnv(gym.Env, Environment):
    def __init__(self, env_config):
        meta_sampler = DynamicWindowSampler()
        self.Environment.__init__(sampler=meta_sampler)

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=torch.float32
        )

    def reset(self):
        return Environment.reset()

    def step(self, action):
        return Environment.step(action)


if __name__ == "__main__":
    args = argsies()

    # Build a Algorithm object from the config and run 1 training iteration.
    # algo = ppo.PPO(env=MetaEnv, config={"num_obs_elements": args.num_obs_elements})
    # algo.train()
