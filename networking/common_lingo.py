from enum import Enum, auto
from typing import Dict

from sampleddetection.datastructures import Action, State
from sampleddetection.utils import unusable

# Nothing much else to add
NetAction = Action
NetState = State


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
