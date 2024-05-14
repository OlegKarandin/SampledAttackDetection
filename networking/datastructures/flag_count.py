from typing import List, Tuple

from sampleddetection.datastructures.context.packet_direction import PacketDirection
from sampleddetection.datastructures.packet_like import PacketLike
from sampleddetection.utils import FLAGS_TO_VAL


class FlagCount:
    """This class extracts features related to the Flags Count."""

    def __init__(self, packets: List[Tuple[PacketLike, PacketDirection]]):
        # self.feature = feature
        self.packets = packets
        self.flags = {
            "F": "FIN",
            "S": "SYN",
            "R": "RST",
            "P": "PSH",
            "A": "ACK",
            "U": "URG",
            "E": "ECE",
            "C": "CWR",
        }

    def has_flag(self, flag, packet_direction=None) -> bool:
        """Count packets by direction.

        Returns:
            packets_count (int):

        """
        packets = (
            (
                packet
                # for packet, direction in self.feature.packets
                for packet, direction in self.packets
                if direction == packet_direction
            )
            if packet_direction is not None
            else (packet for packet, _ in self.packets)
        )

        for packet in packets:
            # TODO:  Assumign this flow is of a single protocol then this is redundant and we could just check the first packet
            if "TCP" in packet:
                if packet.flags[flag]:
                    return True
        return False

    def count(self, flag: str, packet_direction=None) -> int:
        """Count packets by direction.

        Returns:
            packets_count (int):

        """
        packets = (
            (
                packet
                for packet, direction in self.packets
                if direction == packet_direction
            )
            if packet_direction is not None
            else (packet for packet, _ in self.packets)
        )
        # NOTE: The line below was `packet.flags`  but it seems like flags is not the appropriate field

        count = 0
        for packet in packets:
            # TODO:  Assumign this flow is of a single protocol then this is redundant and we could just check the first packet
            if "TCP" in packet:
                if packet.flags[flag]:
                    count += 1
        return count
