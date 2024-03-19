import os

# Add parent dir to include dir
import sys
from pathlib import Path

import pytest

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(f"Parent path is {parent_path}")
sys.path.append(parent_path)

from sampleddetection.samplers.window_sampler import (
    DynamicWindowSampler,
    UniformWindowSampler,
)


def test_flowbytes(csvfile):
    csvpath = Path(csvfile)
    print(f"We will work with csv file {csvfile}")
    dynamic_window_sampler = DynamicWindowSampler(csvpath)
    assert 1 == 1


@pytest.fixture
def csvfile(request):
    return request.config.getoption("--csvfile")
