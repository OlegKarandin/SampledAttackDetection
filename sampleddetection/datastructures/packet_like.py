from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

# The order in which I stored them in the file
# TODO:: Remove this hardcoded danger.
order = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]


class PacketLike(ABC):
    @property
    @abstractmethod
    def time(self) -> float:
        pass

    @property
    @abstractmethod
    def flags(self):
        """
        See `pcap_to_csv.py` file to understand the order
        Should be ["TCP", "FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]
        """
        pass

    @abstractmethod
    def __contains__(self, protocol: str):
        pass


class CSVPacket(PacketLike):
    def __init__(self, row: pd.Series):
        self.row: pd.Series = row
        self.layers = row["layers"]

    @property
    def time(self) -> float:
        return self.row["timestamp"]  # type:ignore

    @property
    def flags(self) -> Dict[str, bool]:
        boolean_list = {o: self.row["flags_mask"][i] for i, o in enumerate(order)}
        return boolean_list  # type: ignore
        # take the list of booleans

    def __contains__(self, protocol: str):
        return protocol in self.row["layers"]
