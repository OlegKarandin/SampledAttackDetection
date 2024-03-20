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
    logger.info(f"First and last sniffs are {first_sniff} -> {final_sniff}")
    non_existant_window = 1e-16
    flowsesh = dws.sample(
        first_sniff,
        non_existant_window,
        final_sniff - first_sniff + non_existant_window + 1,
    )
    logger.info(f"Done with sampling. Getting data from sample ")
    data = flowsesh.get_data()
    sfp = next(iter(flowsesh.flows.values())).packets  # single flowpackets
    logger.info(f"sfp son is of type {type(sfp[0])}")
    pretty_packets = json.dumps([s[0].row.to_dict() for s in sfp], indent=4)
    pretty_str = json.dumps(
        {str(t): v for t, v in data.items()}, indent=4, cls=NpEncoder
    )
    logger.debug(f"Raw data dawg: \n{pretty_str}")
    logger.debug(f"Packets for this flow are:\n{pretty_packets}")
    first_key = next(iter(data))
    logger.info(f"First key is {first_key}")
    bytes_length = data[first_key]["flow_byts_s"]
    print(f"Data is \n{data}")
    assert bytes_length == 142114  # DATA extracted from wireshark


@pytest.fixture
def csvreader(request) -> CSVReader:
    return CSVReader(Path(request.config.getoption("--csvfile")))
