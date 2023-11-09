class FlagCount:
    """This class extracts features related to the Flags Count."""

    def __init__(self, feature):
        self.feature = feature
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

    # NOTE: The original version of this method did not check if packet is "TCP" before getting flags
    def has_flag(self, flag, packet_direction=None) -> bool:
        """Count packets by direction.

        Returns:
            packets_count (int):

        """
        packets = (
            (
                packet
                for packet, direction in self.feature.packets
                if direction == packet_direction
            )
            if packet_direction is not None
            else (packet for packet, _ in self.feature.packets)
        )
        # NOTE: The line below was `packet.flags`  but it seems like flags is not the appropriate field

        for packet in packets:
            # TODO:  Assumign this flow is of a single protocol then this is redundant and we could just check the first packet
            if "TCP" in packet:
                if flag[0] in str(packet.flags):
                    return 1
        return 0
