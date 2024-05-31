"""
Trying raylib for the experiments
Likely mostly to serve as a benchmark of an industrial solution
"""

import argparse
import ast
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import ray
import torch
from gymnasium.wrappers.normalize import NormalizeObservation
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

# NOTE: Importing this is critical to load all model automatically.
from gymenvs.explicit_registration import explicit_registration
from networking.common_lingo import Attack
from networking.downstream_tasks.deepnets import Classifier
from networking.netfactories import NetworkFeatureFactory, NetworkSampleFactory
from sampleddetection.datastructures import CSVSample
from sampleddetection.readers import AbstractTimeSeriesReader, CSVReader
from sampleddetection.reward_signals import DNN_RewardCalculator
from sampleddetection.utils import setup_logger


def str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {s}")


@ray.remote
class RemoteCSVReader(CSVReader):
    def __init__(self, csv_path) -> None:
        super().__init__(csv_path)

    def __getitem__(self, idx) -> CSVSample:
        return super().__getitem__(idx)

    def inittime(self) -> float:
        return super().init_time

    def fintime(self) -> float:
        return super().fin_time


# Maybe we can create our own class that depends on Rays Remote
class RemoteReader(AbstractTimeSeriesReader):
    def __init__(self, path: Path):
        # Initialize a different csv reader
        # self.remote_reader_handle = RemoteCSVReader.remote(path)
        # remote_class = ray.remote(CSVReader)
        self.remote_reader_handle = RemoteCSVReader.remote(path)

    def __getitem__(self, idx) -> Any:
        """
        Will get ray promises and wait for them
        """
        future = self.remote_reader_handle.__getitem__.remote(idx)
        return ray.get(future)

    def __len__(self) -> int:
        future = self.remote_reader_handle.__len__.remote()
        return ray.get(future)

    def getTimestamp(self, idx) -> float:
        future = self.remote_reader_handle.getTimestamp.remote(idx)
        return ray.get(future)

    @property
    def init_time(self) -> float:
        future = self.remote_reader_handle.inittime.remote()
        return ray.get(future)

    @property
    def fin_time(self) -> float:
        future = self.remote_reader_handle.fintime.remote()
        return ray.get(future)


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
        num_possible_actions=args.num_possible_actions,
        data_reader=data_reader,
        action_idx_to_direction=args.action_dir,
        sample_factory=sample_factory,
        feature_factory=feature_factory,
        reward_calculator=reward_calculator,
    )
    print("MANAGED TO MAKE NETENV")
    # Use wrapper to normalize the data:
    env = NormalizeObservation(env)
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
    data_reader = RemoteReader(csv_path)

    # Make the environment
    print("Make the environment")
    register_env("WrappedNetEnv", env_wrapper)

    print("Resetting the environment")
    environment_seed = 42
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
