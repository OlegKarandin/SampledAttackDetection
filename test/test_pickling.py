import pickle
import pytest
import os, sys
import pandas as pd

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(f"Parent path is {parent_path}")
sys.path.append(parent_path)
from networking.datastructures.packet_like import CSVPacket


def test_pickleCSV(csv_path: str):
    """
    Ensure that a specific kind of packet can be pickled before sending through IPC
    """
    csv = pd.read_csv(csv_path)
    packet = CSVPacket(csv.iloc[0])
    pickled_obj = pickle.dumps(packet)
    unpickled_obj = pickle.loads(pickled_obj)
    assert isinstance(
        unpickled_obj, CSVPacket
    ), "Unpickled object is not an instance of CSVPacket"


@pytest.fixture
def csv_path(request) -> str:
    csvreader = request.config.getoption("--mini_csv_path")
    return csvreader
