"""
@inspiration: https://link.springer.com/article/10.1007/s10922-021-09633-5

This script takes idea of @inspiration and tries to see if can detect
presence of attack with a smaller amount of data.
"""
import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List

import debugpy
import joblib
import numpy as np
import sklearn.metrics as mt
import torch
from tqdm import tqdm

from sampleddetection.rl.model import Environment
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
        required=False
    )
    argparsr.add_argument("--csv_path", )
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

    # Warn that the pcap_path argument is deprecated and should not be used unless sure
    if "--pcap_path" in argparsr.parse_args():
        logger.warn(
            "🛑 WARNING: The --pcap_path argument is deprecated and should not be used unless you are sure of what you are doing."
        )


    return argparsr.parse_args()


def accuracy_metrics(y_test, y_pred):
    return {
        "accuracy": round(mt.accuracy_score(y_test, y_pred), 2),  # type: ignore
        "precision": round(mt.precision_score(y_test, y_pred, labels=[1]), 2),
        "recall": round(mt.recall_score(y_test, y_pred, labels=[1]), 2),
        "f1score": round(mt.f1_score(y_test, y_pred, labels=[1]), 2),
    }


def sanitize_args(args: argparse.Namespace):
    assert Path(args.pcap_path).exists(), "Path provided does not exist."
    args.pcap_path = Path(args.pcap_path)
    # TODO: More stuff for window size and the like
    return args


def training_loop(
    environment: Environment,
    episode_length: int,
    episodes: int = 1000,
):
    """
    Problems to solve. How to
    """
    # Start Environment
    # episodes_bar = tqdm(range(episodes), desc="Training over episodes")
    for episode in range(episodes):
        cur_state = environment.reset(torch.Tensor([1500000000]))
        exit()  # TOREM: once `reset()` works well
        # Get Initial State

    # Pick a place to start at random.

    # Just let the model decide its own training-rate paradigm.


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

    features_names = [
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Mean",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Mean",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
    ]

    # %% Import Data
    # ala ./scripts/cc2018_dataset_parser.py
    # window_iterator = PCapWindowItr(
    # path=args.pcap_path,
    # window_skip=args.window_skip,  # Seconds
    # window_length=args.window_skip,
    # )
    scale_iterations = np.logspace(-1, 4, 10, dtype=float)
    logger.info(f"Using iterations f{scale_iterations}")

    ##############################
    # Load Pre-Trained Model
    ##############################
    # CHECK: I't can't be a good idea to use a model trained differently for testing here
    # model = joblib.load(args.model_path)

    ##############################
    # Run Simulation
    ##############################
    dynamic_sampler = DynamicWindowSampler(args.)
    environment = Environment(dynamic_sampler, 2)
    training_loop(environment, 12)

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
