from abc import ABC, abstractmethod
from typing import Dict, List

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
    def flags(self) -> List[str]:
        """
        See `pcap_to_csv.py` file to understand the order
        Should be ["TCP", "FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]
        """
        pass

    @property
    @abstractmethod
    def src_ip(self) -> str:
        pass

    @property
    @abstractmethod
    def dst_ip(self) -> str:
        pass

    @property
    @abstractmethod
    def src_port(self) -> int:
        pass

    @property
    @abstractmethod
    def dst_port(self) -> int:
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

    @property
    def src_ip(self) -> str:
        return self.row["src_ip"]  # type: ignore

    @property
    def dst_ip(self) -> str:
        return self.row["dst_ip"]  # type: ignore

    @property
    def src_port(self) -> str:
        return self.row["src_port"]  # type: ignore

    @property
    def dst_port(self) -> str:
        return self.row["dst_port"]  # type: ignore

    def __contains__(self, protocol: str):
        return protocol in self.row["layers"]
