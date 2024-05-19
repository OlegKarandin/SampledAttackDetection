# Import Enum and Make an Enum out of all possible attacks from the CIC IDC 2017 dataset
# This will be used to determine the possible actions for the RL agent.
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Sequence, Union

import numpy as np
from scapy.plist import PacketList

from sampleddetection.utils import unusable

ArrayLike = Union[np.ndarray, Sequence[Union[int, float, Sequence[Any]]]]


@dataclass
class TimeWindow:
    start: float
    end: float


@unusable(reason="Just never got used", date="Mar 29, 2024")
class RelevantStats(NamedTuple):
    fwd_Packet_Length_Max: int
    fwd_Packet_Length_Min: int
    fwd_Packet_Length_Mean: int
    bwd_Packet_Length_Max: int
    bwd_Packet_Length_Min: int
    bwd_Packet_Length_Mean: int
    flow_Bytess: int
    flow_Packets: int
    flow_IAT_Mean: float
    flow_IAT_Max: float
    flow_IAT_Min: float
    fwd_IAT_Mean: float
    fwd_IAT_Max: float
    fwd_IAT_Min: float
    bwd_IAT_Mean: float
    bwd_IAT_Maxs: float
    bwd_IAT_Min: float
    min_Packet_Length: int
    max_Packet_Length: int
    packet_Length_Mean: float

    @staticmethod
    def create(packet_list: PacketList):
        # Form all flows
        flow_dict: Dict[FlowKey, Flow] = {}
        for packet in packet_list:
            flow_key = FlowKey.create(packet)
            # if flow_key not in flow_dict:
            #     flow_dict[flow_key] = Flow.create(flow_key)
            # flow_dict[flow_key].add_packet(packet)
