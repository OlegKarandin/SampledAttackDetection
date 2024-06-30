"""
@sourced from: https://github.com/hieulw/cicflowmeter
"""

import time
from enum import Enum
from typing import Dict, Sequence, Tuple

from networking.common_lingo import ATTACK_TO_STRING
from networking.datastructures.packet_like import PacketLike
from sampleddetection.common_lingo import TimeWindow
from sampleddetection.utils import setup_logger, unusable

from .constants import EXPIRED_UPDATE, GARBAGE_COLLECT_PACKETS
from .context.packet_direction import PacketDirection
from .context.packet_flow_key import get_packet_flow_key
from .flow import Flow


# There shoudl be no need for this inheritance
# class SampledFlowSession(DefaultSession):
class SampledFlowSession:
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows: Dict[Tuple[Tuple, int], Flow] = {}
        # WARN: this can lead to too many files being open. Careful there
        # self.logger = setup_logger(self.__class__.__name__)
        self.packets_count = 0
        self.logger = setup_logger(__class__.__name__)

        # Sample Variables
        # CHECK: If we actually need to use below and its references
        # self.sampwindow_length = kwargs["sampwindow_length"]
        # self.samp_curinitpos = kwargs["sample_initpos"]

        self.time_window = TimeWindow(-1, -1)

    @unusable(
        reason="I dont see the need for exporting packetList", date="Mar 18, 2024"
    )
    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        # self.garbage_collect(None)
        return None

    # def update_sampling_params(self, samp_freq, samp_wind, samp_curinitpos) -> None:
    #     self.sampwindow_length = samp_wind
    #     self.samp_curinitpos = samp_curinitpos

    def on_packet_received(self, packet: PacketLike):
        """
        Return:
        -------
            - finished_state: whether or not we have met the end of the sampling window.

        Assumption:
        -----------
        Outer function will make sure we have not met the end of the sampling window.
        """

        _init_time = time.time()
        assert isinstance(
            packet, PacketLike
        ), "Assertion Error: Packet received is expected to be of type `PacketLike`"

        count = 0
        direction = PacketDirection.FORWARD

        # We will not deal with non TCP/UDP protocols for now
        # CHECK:
        if not ("TCP" in packet) and not ("UDP" in packet):
            return False

        # Ensure that we are not taking sample outside our window

        # Creates a key variable to check
        # self.logger.debug(
        #     f"Packet with timestamp {packet.time} look for key going forward"
        # )
        try:
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return False

        self.packets_count += 1
        _time_get_key = time.time()
        # If there is no forward flow with a count of 0
        # self.logger.debug(
        #     f"Packet with timestamp {packet.time} key found. Looking in dict"
        # )
        if flow is None:

            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
            # self.logger.debug(
            #     f"Packet with timestamp {packet.time} not found forward flow maybe "
            # )

        if flow is None:
            # If no flow exists create a new flow
            direction = PacketDirection.FORWARD
            # self.logger.debug(
            #     f"Packet with timestamp {packet.time} not found in any direction. Trying to create"
            # )
            flow = Flow(packet, direction)
            packet_flow_key = get_packet_flow_key(packet, direction)
            self.flows[(packet_flow_key, count)] = flow
            # self.logger.warn(
            #     f"You have added a new flow with the tuple : {(packet_flow_key, count)}"
            # )

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            # If the packet exists in the flow but the packet is sent
            # after too much of a delay than it is a part of a new flow.
            expired = EXPIRED_UPDATE
            # self.logger.debug(
            #     f"Packet with timestamp {packet.time} checking if expired."
            # )
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                # self.logger.debug(
                #     f"\t With packet.time {packet.time}, flow.latest_timestamp {flow.latest_timestamp} "
                #     f"and resulting count {count}"
                # )
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    # self.logger.debug(
                    #     f"\t\t Found our whole at {packet}, {direction} and count {count}"
                    #     f"and resulting count {count}"
                    # )
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break

        _deciding_time = time.time()

        # Update time_window
        self.time_window.start = min(self.time_window.start, packet.time)
        self.time_window.end = max(self.time_window.end, packet.time)

        # Finally add_packet
        _min_max_time = time.time()
        flow.add_packet(packet, direction)
        # self.logger.debug(f"SampledFlowSession adding packet {packet.time}")
        _add_flow_end_time = time.time()

        fin_time = time.time()
        # self.logger.debug(
        #     f"on_packet_received takes {fin_time - _init_time:.3e} subtimes are:\n"
        #     f"time_get_key : {_time_get_key-_init_time:.3e}\n"
        #     f"_deciding_time : {_deciding_time - _time_get_key:.3e}\n"
        #     f"_min_max_time : {_min_max_time - _deciding_time:.3e}\n"
        #     f"Adding packet time : {_add_flow_end_time - _min_max_time:.3e}\n"
        # )
        return True

    def flow_label_distribution(self) -> Dict[Enum, int]:
        labels = {enum: 0 for enum, label in ATTACK_TO_STRING.items()}
        for _, flow in self.flows.items():
            flabel = flow.label
            labels[flabel] += 1
        return labels

    # property
    def num_flows(self) -> int:
        return len(self.flows)

    def _finish_flow(self, packet_time, latest_time) -> None:
        pass  # TODO:

    def get_data(self) -> Dict[Tuple, Dict]:
        info = {}
        # self.logger.info(f"We have {len(self.flows)} to go through")
        if len(self.flows) > 0:
            for k, v in self.flows.items():
                info[k] = v.get_data()
        return info

    def get_packets(self) -> Sequence[PacketLike]:
        packets = []
        for _, flow in self.flows.items():
            packets.extend(flow.packets)
        return packets

    def reset(self):
        """
        Clears all state in this SampledFlowSession
        """
        self.flows.clear()
        self.packets_count = 0
        self.time_window = TimeWindow(-1, -1)


@unusable(reason="No longer using scapy sniffers", date="< Mar 29, 2024")
def generate_session_class(output_mode, output_file, verbose):
    return type(
        "NewFlowSession",
        (FlowSession,),  # type: ignore
        {
            "output_mode": output_mode,
            "output_file": output_file,
            "verbose": verbose,
        },
    )
