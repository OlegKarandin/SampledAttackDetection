"""
@sourced from: https://github.com/hieulw/cicflowmeter
"""

from typing import Dict, Tuple, Union

from scapy.packet import Packet
from scapy.sessions import DefaultSession

from sampleddetection.datastructures.packet_like import PacketLike
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

        # Sample Variables
        # TODO: Ensure these things are being passed
        self.sampwindow_length = kwargs["sampwindow_length"]
        self.samp_curinitpos = kwargs["sample_initpos"]

    @unusable(
        reason="I dont see the need for exporting packetList", date="Mar 18, 2024"
    )
    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        # self.garbage_collect(None)
        return None

    def update_sampling_params(self, samp_freq, samp_wind, samp_curinitpos) -> None:
        self.sampwindow_length = samp_wind
        self.samp_curinitpos = samp_curinitpos

    def on_packet_received(self, packet: PacketLike):
        """
        Return:
        -------
            finished_state: whether or not we have met the end of the sampling window.
        Assumption:
        -----------
        Outer function will make sure we have not met the end of the sampling window.
        """

        count = 0
        direction = PacketDirection.FORWARD

        # We will not deal with non TCP/UDP protocols for now
        # CHECK:
        if not ("TCP" in packet) and not ("UDP" in packet):
            return False

        # Ensure that we are not taking sample outside our window

        # Creates a key variable to check
        try:
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return False

        self.packets_count += 1

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

        if flow is None:
            # If no flow exists create a new flow
            direction = PacketDirection.FORWARD
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
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break

        flow.add_packet(packet, direction)

        return True

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


def generate_session_class(output_mode, output_file, verbose):
    return type(
        "NewFlowSession",
        (FlowSession,),
        {
            "output_mode": output_mode,
            "output_file": output_file,
            "verbose": verbose,
        },
    )
