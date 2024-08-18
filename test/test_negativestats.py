"""
All the hardcoded statistics used in this script where 
calculated by hand.
"""

import ast
import os

# Add parent dir to include dir
import sys
from pathlib import Path

import pandas as pd
import pytest


import random
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(f"Parent path is {parent_path}")
sys.path.append(parent_path)

from logging import DEBUG as LVL_DEBUG

from networking.common_lingo import Attack
from networking.readers import NetCSVReader
from sampleddetection.utils import setup_logger
from sampleddetection.environments import SamplingEnvironment
from networking.datastructures.flowsession import SampledFlowSession
from networking.netfactories import NetworkSampleFactory
from networking.samplers import WeightedSampler

global logger
logger = setup_logger(os.path.basename(__file__), LVL_DEBUG)

def test_negetive_statistics(csv_reader: NetCSVReader, random_draws: int):
    labels = [
        Attack.BENIGN,
        Attack.HULK,
        Attack.GOLDENEYE,
        Attack.SLOWLORIS,
        Attack.SLOWHTTPTEST,
    ]
    global_sampler = WeightedSampler(
        csv_reader,
        12,
        1600,
        labels,
        binary_labels=True,
    )

    wskip_range = SamplingEnvironment.WINDOW_SKIP_RANGE
    wl_range = SamplingEnvironment.WINDOW_LENGTH_RANGE

    first_sniff_time, last_sniff_time = csv_reader.first_sniff_time, csv_reader.last_sniff_time.

    # Draw any one of those at random and sample
    for _ in range(random_draws):
        # Draw at randomly and wait for a failure
        random_pos = random.random() * (last_sniff_time - first_sniff_time) + first_sniff_time
        random_ws = random.random() * (wskip_range[1] - wskip_range[0]) + wskip_range[0]
        random_wl = random.random() * (wl_range[1] - wl_range[0]) + wl_range[0]
        # Lets get ths samples
        samples  = global_sampler.sample(random_pos, random_ws, random_wl, first_sample=True)
        flowsesh = SampledFlowSession()
        flowsesh.reset()

        #def binary_search(cpt_rdr: AbstractTimeSeriesReader, target_time: float) -> int:



        # Check that the return packages are in order

@pytest.fixture
def flowsesh(request) -> NetCSVReader:
    csv_path = Path(request.config.getoption("--csvfile"))
    csv_reader = NetCSVReader(csv_path)
    return csv_reader


@pytest.fixture
def csv_path(request) -> str:
    csvreader = request.config.getoption("--csvfile")
    return csvreader

@pytest.fixture
def pcap_path(request) -> str:
    pcappath = request.config.getoption("--pcappath")
    return pcappath

@pytest.fixture
def active_timeout(request) -> int:
    val = request.config.getoption("--active_timeout")
    return val

@pytest.fixture
def clump_timeout(request) -> int:
    val = request.config.getoption("--clump_timeout")
    return val
def random_draws(request) -> int:
    val = request.config.getoption("--draws")
    return val
