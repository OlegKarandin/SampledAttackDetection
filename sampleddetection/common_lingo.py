# Import Enum and Make an Enum out of all possible attacks from the CIC IDC 2017 dataset
# This will be used to determine the possible actions for the RL agent.
from enum import Enum, auto
from typing import Dict, List, NamedTuple

from scapy.plist import PacketList

from sampleddetection.utils import unusable


class Attack(Enum):
    BENIGN = auto()
    SLOWLORIS = auto()
    SLOWHTTPTEST = auto()
    HULK = auto()
    GOLDENEYE = auto()
    HEARTBLEED = auto()
    # CHECK: Not so sure about these ones below
    WEB_BRUTEFORCE = auto()
    XSS = auto()
    SQL_INJECTION = auto()
    DROPBOX = auto()
    COOLDISK = auto()
    PORTSCAN_NMAPP = auto()
    BOTNET_ARES = auto()
    PORT_SCAN = auto()
    DDOS_LOIT = auto()
    GENERAL = auto()


# Create a Dictionary to labels `./data/label_data.json`
ATTACK_TO_STRING: Dict[Enum, str] = {
    Attack.BENIGN: "Benign",
    Attack.SLOWLORIS: "DoS_Slowloris",
    Attack.HULK: "DoS_Hulk",
    Attack.SLOWHTTPTEST: "DoS_Slowhttptest",
    Attack.GOLDENEYE: "DoS_GoldenEye",
    Attack.HEARTBLEED: "HeartBleed",
    # TODO: (SUPER) : Fill the later with universal labels
    Attack.WEB_BRUTEFORCE: "WEB_BRUTEFORCE",
    Attack.XSS: "XSS",
    Attack.SQL_INJECTION: "SQL_INJECTION",
    Attack.DROPBOX: "DROPBOX",
    Attack.COOLDISK: "COOLDISK",
    Attack.PORTSCAN_NMAPP: "PORTSCAN_NMAPP",
    Attack.BOTNET_ARES: "BOTNET_ARES",
    Attack.PORT_SCAN: "PORT_SCAN",
    Attack.DDOS_LOIT: "DDOS_LOIT",
    Attack.GENERAL: "General",
}

STRING_TO_ATTACKS: Dict[str, Attack] = {v: k for k, v in ATTACK_TO_STRING.items()}  # type: ignore


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
