"""
Trying raylib for the experiments
Likely mostly to serve as a benchmark of an industrial solution
"""

import argparse
import ast
import json
from argparse import ArgumentParser
from typing import Dict

import gymnasium as gym
import torch
from ray.rllib.algorithms import ppo

import gymenvs


def str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {s}")


def argsies():
    ap = ArgumentParser()
    ap.add_argument(
        "--csv_path_str",
        default="./data/mini_wednesday.csv",
        type=str,
        help="Path to where the data lies",
    )
    ap.add_argument(
        "--num_possible_actions",
        default=2,
        type=int,
        help="Dimension for action vector",
    )
    ap.add_argument(
        "--paradigm_constants",
        default="./paradigm_constants.json",
        type=str,
        help="Where training/paradigm constants get stored.",
    )

    # Prelimns
    ap.add_argument(
        "--action_idx_to_direction",
        default="{0: -1, 1: 1}",
        type=str_to_dict,
        help="Map between indices outputted by model vs values they actually represent. ",
    )

    # Parse the argument
    args = ap.parse_args()

    # Check on their values
    ## Add Values Manually
    with open(args.paradigm_constants, "r") as f:
        # Add the amount of observations.
        paradigm_spec_file = json.load(f)
        desired_features = paradigm_spec_file["desired_features"]
        args.obs_elements = desired_features

        # Add Actions
        actions = paradigm_spec_file["actions"]
        action_dir = {i: a for i, a in enumerate(actions)}
        args.action_dir = action_dir

    return args


if __name__ == "__main__":
    args = argsies()

    # gymenvs.register_env()

    # Make the environment
    print("Make the environment")
    env = gym.make(
        "NetEnv-v0",
        csv_path_str=args.csv_path_str,
        num_obs_elements=len(args.obs_elements),
        num_possible_actions=args.num_possible_actions,
        action_idx_to_direction=args.action_dir,
    )

    print("Resetting the environment")
    environment_seed = 42
    env.reset()

    # Build a Algorithm object from the config and run 1 training iteration.
    # algo = ppo.PPO(env=MetaEnv, config={"num_obs_elements": args.num_obs_elements})
    # algo.train()
