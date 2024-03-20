import json
import os

# Add parent dir to include dir
import sys
from pathlib import Path

import pandas as pd
import pytest

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(f"Parent path is {parent_path}")
sys.path.append(parent_path)

from logging import DEBUG as LVL_DEBUG
from logging import INFO as LVL_INFO

from sampleddetection.datastructures.flowsession import SampledFlowSession
from sampleddetection.utils import NpEncoder, setup_logger

global logger
logger = setup_logger(os.path.basename(__file__), LVL_DEBUG)

from sampleddetection.readers.readers import CSVReader
from sampleddetection.samplers.window_sampler import (
    DynamicWindowSampler,
    UniformWindowSampler,
)


def test_flowbytes(csvreader: CSVReader):
    logger.info(f"We will work with csv file {csvreader.csv_path}")
    dws = DynamicWindowSampler(csvrdr=csvreader)
    first_sniff = dws.csvrdr.first_sniff_time - 1
    final_sniff = dws.csvrdr.last_sniff_time
    non_existant_window = 1e-16
    flowsesh = dws.sample(
        first_sniff,
        non_existant_window,
        final_sniff - first_sniff + non_existant_window + 1,
    )
    data = flowsesh.get_data()
    first_key = next(iter(data))
    bytes_length = (
        data[first_key]["totlen_fwd_pkts"] + data[first_key]["totlen_bwd_pkts"]
    )
    assert bytes_length == 142114  # DATA extracted from wireshark


@pytest.fixture
def csvreader(request) -> CSVReader:
    return CSVReader(Path(request.config.getoption("--csvfile")))


# @pytest.fixture
# def sample(request) -> SampledFlowSession:
#     src_ip =
#     return CSVReader(
