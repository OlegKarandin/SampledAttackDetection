"""
Trying raylib for the experiments
Likely mostly to serve as a benchmark of an industrial solution
"""

import random
import argparse
import ast
import json
from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np

import gymnasium as gym
import ray
from gymnasium.wrappers.normalize import NormalizeObservation
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

# NOTE: Importing this is critical to load all model automatically.
from gymenvs.explicit_registration import explicit_registration
from networking.common_lingo import Attack
from networking.downstream_tasks.deepnets import Classifier
from networking.netfactories import NetworkFeatureFactory, NetworkSampleFactory
from networking.readers import NetCSVReader
from sampleddetection.reward_signals import DNN_RewardCalculator
from sampleddetection.utils import setup_logger


def str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {s}")


def argsies():
    ap = ArgumentParser()
    ap.add_argument(
        "--csv_path_str",
        # default="./data/mini_wednesday.csv",
        default="./data/Wednesday.csv",
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
    ap.add_argument(
        "--sampling_budget",
        default=12,
        type=int,
        help="How many sampling windows between window skips.",
    )
    ap.add_argument(
        "--random_seed",
        default=420,
        type=int,
        help="Seed for random generators",
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

    assert Path(
        args.csv_path_str
    ).exists(), f"--csv_path_str {args.csv_path_str} does not exist."

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


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def env_wrapper(env) -> gym.Env:
    # Call the registration
    explicit_registration()

    # Specify the NetworkSampleFactor
    sample_factory = NetworkSampleFactory()
    feature_factory = NetworkFeatureFactory(args.obs_elements, attacks_to_detect)

    num_features = len(args.obs_elements)

    # Create Data Reader
    # csv_path = Path(args.csv_path_str)
    # assert csv_path.exists(), "csv path provided does not exist"
    # data_reader = CSVReader(csv_path)

    # Create the downstream classidication learner
    classifier = Classifier(
        input_size=num_features, output_size=len(attacks_to_detect) + 1
    )
    # Create reward calculator to use
    reward_calculator = DNN_RewardCalculator(classifier)

    print("Trying to make NETENVE")
    env = gym.make(
        "NetEnv",
        num_obs_elements=len(args.obs_elements),
        actions_max_vals=Action(60, 10),
        data_reader_ref=csv_reader_ref,
        action_idx_to_direction=args.action_dir,
        sample_factory=sample_factory,
        feature_factory=feature_factory,
        reward_calculator=reward_calculator,
        sampling_budget=args.sampling_budget,
    )
    print("MANAGED TO MAKE NETENV")
    # Use wrapper to normalize the data:
    # env = NormalizeObservation(env)
    return env


if __name__ == "__main__":
    args = argsies()

    # Make the logger
    logger = setup_logger(__name__)
    logger.info("Starting main part of script.")
    # gymenvs.register_env()

    # Define which labels one expects on the given dataset
    attacks_to_detect = [
        Attack.SLOWLORIS,
        Attack.SLOWHTTPTEST,
        Attack.HULK,
        Attack.GOLDENEYE,
        # Attack.HEARTBLEED. # Takes too long find in dataset.
    ]

    # Columns to Normalize
    columns_to_normalize = [
        "fwd_pkt_len_max",
        "fwd_pkt_len_min",
        "fwd_pkt_len_mean",
        "bwd_pkt_len_max",
        "bwd_pkt_len_min",
        "bwd_pkt_len_mean",
        "flow_byts_s",
        "flow_pkts_s",
        "flow_iat_mean",
        "flow_iat_max",
        "flow_iat_min",
        "fwd_iat_mean",
        "fwd_iat_max",
        "fwd_iat_min",
        "bwd_iat_max",
        "bwd_iat_min",
        "bwd_iat_mean",
        "pkt_len_min",
        "pkt_len_max",
        "pkt_len_mean",
    ]
    # Make shared CSVReader
    csv_path = Path(args.csv_path_str)
    csv_reader = NetCSVReader(csv_path)
    csv_reader_ref = ray.put(csv_reader)

    # Make the environment
    print("Make the environment")
    register_env("WrappedNetEnv", env_wrapper)

    print("Resetting the environment")
    set_all_seeds(args.random_seed)
    algo = (
        PPOConfig()
        .env_runners(num_env_runners=1)
        .resources(num_gpus=0)
        .environment(env="WrappedNetEnv")
        .build()
    )

    for i in range(20):
        result = algo.train()
        print(pretty_print(result))

    # Build a Algorithm object from the config and run 1 training iteration.
    # algo = ppo.PPO(env=MetaEnv, config={"num_obs_elements": args.num_obs_elements})
    # algo.train()
