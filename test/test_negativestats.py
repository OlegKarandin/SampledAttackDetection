"""
All the hardcoded statistics used in this script where 
calculated by hand.
"""

import os
import random

# Add parent dir to include dir
import sys
from pathlib import Path

import pytest

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(f"Parent path is {parent_path}")
sys.path.append(parent_path)

from logging import DEBUG as LVL_DEBUG

from networking.common_lingo import Attack
from networking.datastructures.flowsession import SampledFlowSession
from networking.readers import NetCSVReader
from networking.samplers import WeightedSampler
from networking.datastructures.context.packet_direction import PacketDirection
from sampleddetection.environments import SamplingEnvironment
from sampleddetection.utils import setup_logger

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

    first_sniff_time, last_sniff_time = csv_reader.first_sniff_time, csv_reader.last_sniff_time

    # Draw any one of those at random and sample
    for d in range(random_draws):
        # Draw at randomly and wait for a failure
        random_pos = random.random() * (last_sniff_time - first_sniff_time) + first_sniff_time
        random_ws = random.random() * (wskip_range[1] - wskip_range[0]) + wskip_range[0]
        random_wl = random.random() * (wl_range[1] - wl_range[0]) + wl_range[0]

        # Sample
        samples  = global_sampler.sample(random_pos, random_ws, random_wl)

        # Sometimes we may get empty samples
        if len(samples) == 0:
            continue

        # Test to ensure continuity of samples
        for i in range(1,len(samples)):
            assert samples[i].time >= samples[i-1].time


        # Test to ensrue negative statistics
        flowsesh = SampledFlowSession()
        for s in samples:
            flowsesh.on_packet_received(s)
        # Now Retrieve the statistics
        flowsesh.get_data() # Will throw an error if samples not in order


@pytest.fixture
def csv_reader(request) -> NetCSVReader:
    csv_path = Path(request.config.getoption("--complete_compressed"))
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

@pytest.fixture
def random_draws(request) -> int:
    val = request.config.getoption("--draws")
    return val
