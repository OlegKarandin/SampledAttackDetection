"""
@inspiration: https://link.springer.com/article/10.1007/s10922-021-09633-5

This script takes idea of @inspiration and tries to see if can detect
presence of attack with a smaller amount of data.
"""
import argparse
import logging
from pathlib import Path
from typing import List

import debugpy
import numpy as np
import sklearn.metrics as mt
import torch

from sampleddetection.common_lingo import Action, State
from sampleddetection.environment.agents import AgentLike, RLAgent
from sampleddetection.environment.model import Environment
# %% Import parsers
from sampleddetection.samplers.window_sampler import (DynamicWindowSampler,
                                                      UniformWindowSampler)
from sampleddetection.statistics.window_statistics import (flow_to_stats,
                                                           get_flows)
from sampleddetection.utils import setup_logger

# Set up all random seeds to be the same
logger = setup_logger("MAIN", logging.INFO)


def get_args() -> argparse.Namespace:
    argparsr = argparse.ArgumentParser()
    argparsr.add_argument(
        "--pcap_path",
        type=str,
        help="(Deprecated use csv_path instead) Path to the .pcap file",
        required=False,
    )
    argparsr.add_argument(
        "--csv_path",
        default="./bigdata/Wednesday.csv",
        type=str,
        help="Path to the csv file to read.",
    )
    argparsr.add_argument(
        "--window_skip", type=float, default=1.0, help="Time to skip between windows"
    )
    argparsr.add_argument(
        "--window_length", type=float, default=1.0, help="Length of each window"
    )
    argparsr.add_argument(
        "--model_path",
        default="./models/detection_model.joblib",
        help="Length of each window",
    )
    argparsr.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
    )
    argparsr.add_argument(
        "--features",
        nargs="+",
        default=[
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
        ],
        help="Which statistics to give the model to work with.",
    )

    # Warn that the pcap_path argument is deprecated and should not be used unless sure

    return argparsr.parse_args()


def accuracy_metrics(y_test, y_pred):
    return {
        "accuracy": round(mt.accuracy_score(y_test, y_pred), 2),  # type: ignore
        "precision": round(mt.precision_score(y_test, y_pred, labels=[1]), 2),
        "recall": round(mt.recall_score(y_test, y_pred, labels=[1]), 2),
        "f1score": round(mt.f1_score(y_test, y_pred, labels=[1]), 2),
    }


def sanitize_args(args: argparse.Namespace):
    # assert Path(args.pcap_path).exists(), "Path provided does not exist."
    if "--pcap_path" in args:
        logger.warn(
            "🛑 WARNING: The --pcap_path argument is deprecated and should not be used unless you are sure of what you are doing."
        )
    if "csv_path" in args:
        assert Path(args.csv_path).exists(), "Provided csv file does not exists"
        args.csv_path = Path(args.csv_path)
    # TODO: More stuff for window size and the like
    return args


def training_loop(
    environment: Environment,
    episode_length: int,
    agent: AgentLike,
    episodes: int = 1000,
):
    """
    Problems to solve. How to
    """
    logger = setup_logger("Train_Loop", logging.INFO)
    # DEBUG:
    # pd_lowest_time = environment.sampler.csvrdr.csv_df["timestamp"].min()
    # logger.info(f"Lowest by pandas column is {pd_lowest_time}")

    # CHECK: I dont know why this is the only way to not lose precision but be careful
    lowest_time = torch.zeros([environment.M], dtype=torch.float64)
    lowest_time[0] = environment.sampler.csvrdr[0].time
    logger.info(f"Starting the sampling with the lowest time {lowest_time}")
    # Start Environment
    # episodes_bar = tqdm(range(episodes), desc="Training over episodes")
    init_states: List[State] = environment.reset(lowest_time)
    for episode in range(episodes):
        # Form Tensor Observation

        logger.info("Done with one episode")
        exit()  # TOREM: once `reset()` works well
        # Get Initial State

    # Pick a place to start at random.

    # Just let the model decide its own training-rate paradigm.


def test_mechanism(environment):
    # We just sample from the day for a specific time skip and time 
    


if __name__ == "__main__":
    ##############################
    # Init Values
    ##############################

    args = sanitize_args(get_args())

    if args.debug:
        logger.info("Waiting for client to connect to port 42019")
        debugpy.listen(42019)
        debugpy.wait_for_client()
        logger.info("Client connected, debugging...")

    # TOREM:
    # %% Import Data
    # ala ./scripts/cc2018_dataset_parser.py
    # window_iterator = PCapWindowItr(
    # path=args.pcap_path,
    # window_skip=args.window_skip,  # Seconds
    # window_length=args.window_skip,
    # )

    scale_iterations = np.logspace(-6, 1, 3, dtype=float)
    logger.info(f"Using iterations f{scale_iterations}")
    logger.info(f"Model will be shown the following features {args.features}")

    ##############################
    # Load Pre-Trained Model
    ##############################
    # CHECK: I't can't be a good idea to use a model trained differently for testing here
    # model = joblib.load(args.model_path)

    ##############################
    # Run Simulation
    ##############################
    logger.info(f"Working with file {args.csv_path}")
    dynamic_sampler = DynamicWindowSampler(args.csv_path)
    environment = Environment(dynamic_sampler, 1)
    agent = RLAgent()
    training_loop(environment, 12, agent)

    exit()

    # %% Create Statistical Windows
    limiting_counter = 0

    performance_matrix = np.empty((len(scale_iterations), len(scale_iterations)))

    for window_length in scale_iterations:
        for window_skip in scale_iterations:
            window_iterator = UniformWindowSampler(
                path=args.pcap_path,
                window_skip=window_skip,
                window_length=window_length,
                amount_windows=100,
            )

            # Keep Dictionary of Stats Across Windows
            stat_logger = dict.fromkeys(
                {"accuracy", "precision", "recall", "f1score"}, []
            )

            for window in window_iterator:
                flows = get_flows(window)
                features = flow_to_stats(flows)

                # Clean features
                latent_features = features[features_names]  # type_ignore

                # Get Stats
                # TODO :

                # Pick up the right stats
                # TODO:

                # Add to average
            # performance_matrix[scale_i,scale_j] =
