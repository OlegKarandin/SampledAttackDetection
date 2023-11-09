"""
@inspiration: https://link.springer.com/article/10.1007/s10922-021-09633-5

This script takes idea of @inspiration and tries to see if can detect
presence of attack with a smaller amount of data.
"""
import argparse
import json
import logging
from pathlib import Path

# %% Import parsers
from sampleddetection.parsers.oleg import PCapWindowItr
from sampleddetection.statistics.window_statistics import flow_to_stats, get_flows
from sampleddetection.utils import setup_logger


def get_args() -> argparse.Namespace:
    argparsr = argparse.ArgumentParser()
    argparsr.add_argument(
        "--pcap_path",
        type=str,
        default="./bigdata/Wednesday-WorkingHours.pcap",
        help="Path to the .pcap file",
    )
    argparsr.add_argument(
        "--window_skip", type=float, default=1.0, help="Time to skip between windows"
    )
    argparsr.add_argument(
        "--window_length", type=float, default=1.0, help="Length of each window"
    )

    return argparsr.parse_args()


def sanitize_args(args: argparse.Namespace):
    assert Path(args.pcap_path).exists(), "Path provided does not exist."
    args.pcap_path = Path(args.pcap_path)
    # TODO: More stuff for window size and the like
    return args


##############################
# Init Values
##############################

logger = setup_logger("main", logging.INFO)

args = sanitize_args(get_args())

# %% Import Data
# ala ./scripts/cc2018_dataset_parser.py
window_iterator = PCapWindowItr(
    path=args.pcap_path,
    window_skip=args.window_skip,  # Seconds
    window_length=args.window_skip,
)

##############################
# Load Pre-Trained Model
##############################

##############################
# Run Simulation
##############################

# %% Create Statistical Windows
limiting_counter = 0

for i, window in enumerate(window_iterator):
    logger.info(f"Window has {len(window)} packets available")
    # First print the statistics
    flows = get_flows(window)
    logger.info(f"Obtained {len(flows)} flows in the {i}th window")

    stats = flow_to_stats(flows)
    logger.info(f"📊 Stats as seen below: \n{stats}")

    input("Press Enter to continue...")

    limiting_counter += 1
    if limiting_counter < 3:
        break

print("Done")
