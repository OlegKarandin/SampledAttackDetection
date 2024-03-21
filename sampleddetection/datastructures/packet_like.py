import ast
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
from scapy.all import Packet

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

    @property
    @abstractmethod
    def payload_size(self) -> int:
        pass

    @abstractmethod
    def __contains__(self, protocol: str):
        pass

    @property
    @abstractmethod
    def tcp_window(self) -> int:
        pass

    @property
    @abstractmethod
    def layers(self) -> List[str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def header_size(self) -> int:
        pass

    @property
    @abstractmethod
    def __dict__(self) -> Dict:
        pass


class ScapyPacket(PacketLike):
    """
    Wrapper for a scapy packet with our interface.
    """

    def __init__(self, packet: Packet):
        self.packet: Packet = packet
        self.proto = "None"
        if "UDP" in packet:
            self.proto = "UDP"
        elif "TCP" in packet:
            self.proto = "TCP"

    @property
    def time(self) -> float:
        return float(self.packet.time)

    @property
    def flags(self) -> List[str]:
        return self.packet.sprintf("%TCP.flags%").split("|")

    @property
    def src_ip(self) -> str:
        assert "IP" in self.packet, "ScapyPacket is not IP packet"
        return self.packet["IP"].src

    @property
    def dst_ip(self) -> str:
        assert "IP" in self.packet, "ScapyPacket is not IP packet"
        return self.packet["IP"].dst

    @property
    def src_port(self) -> int:
        return self.packet[self.proto].sport

    @property
    def dst_port(self) -> int:
        return self.packet[self.proto].dport

    @property
    def payload_size(self) -> int:
        if self.proto == "None":
            return 0
        else:
            return self.packet[self.proto].payload

    def __contains__(self, protocol: str):
        return protocol in self.packet

    @property
    def tcp_window(self) -> int:
        assert self.proto == "TCP", f"tcp_window() called on non-tcp packet"
        return self.packet["TCP"].window

    @property
    def layers(self) -> List[str]:
        return [layer.name for layer in self.packet]

    def __len__(self) -> int:
        return len(self.packet)

    @property
    def header_size(self) -> int:
        assert "IP" in self.packet, "ScapyPacket is not IP packet"
        return self.packet["IP"].ihl * 4

    def __dict__(self) -> Dict:
        return {
            "time": self.time,
            "flags": self.flags,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "payload_size": self.payload_size,
            "tcp_window": self.tcp_window,
            "layers": self.layers,
            "header_size": self.header_size,
        }


class CSVPacket(PacketLike):
    def __init__(self, row: pd.Series):
        self.columns = row.index
        self.row: pd.Series = row

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

    @property
    def payload_size(self) -> int:
        return int(self.row["payload_size"])

    @property
    def tcp_window(self) -> int:
        assert "TCP" in self.row["layers"], f"tcp_window() called on non-tcp packet"
        return int(self.row["tcp_window"])

    @property
    def layers(self) -> List[str]:
        return ast.literal_eval(self.row["layers"])

    def __len__(self) -> int:
        return int(self.row["packet_length"])

    @property
    def header_size(self) -> int:
        return int(self.row["int_head_len"])

    def __dict__(self) -> Dict:
        return self.row.to_dict()  # type: ignore
