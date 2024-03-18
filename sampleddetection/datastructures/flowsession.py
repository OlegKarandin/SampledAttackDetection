"""
@sourced from: https://github.com/hieulw/cicflowmeter
"""

from typing import Union

from scapy.packet import Packet
from scapy.sessions import DefaultSession

from sampleddetection.utils import setup_logger

from .constants import EXPIRED_UPDATE, GARBAGE_COLLECT_PACKETS
from .context.packet_direction import PacketDirection
from .context.packet_flow_key import get_packet_flow_key
from .flow import Flow


class SampledFlowSession(DefaultSession):
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.logger = setup_logger(self.__class__.__name__)
        self.packets_count = 0

        # Sample Variables
        # TODO: Ensure these things are being passed
        self.samp_window = kwargs["sampling_window"]
        self.samp_curinitpos = kwargs["sample_initpos"]

        super(SampledFlowSession, self).__init__(*args, **kwargs)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.garbage_collect(None)
        return super(SampledFlowSession, self).toPacketList()

    def update_sampling_params(self, samp_freq, samp_wind, samp_curinitpos) -> None:
        self.samp_window = samp_wind
        self.samp_curinitpos = samp_curinitpos

    def on_packet_received(self, packet: Packet):
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
        if "TCP" not in packet:
            return False
        elif "UDP" not in packet:
            return False

        # Ensure that we are not taking sample outside our window
        packet_time = packet.time
        # if packet.time > samp_left_limit and packet.time < samp_right_limit:
        #     pass
        # else:
        #     return None  # TODO: I think at this point we also have to ensure that the

        # Creates a key variable to check
        try:
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return 0

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

        elif "F" in packet.flags:
            # If it has FIN flag then early collect flow and continue
            flow.add_packet(packet, direction)
            self.garbage_collect(packet.time)
            return True

        if self.packets_count % GARBAGE_COLLECT_PACKETS == 0 or flow.duration > 120:
            self.garbage_collect(packet.time)

    def get_flows(self) -> list:
        return self.flows.values()

    # property
    def num_flows(self) -> int:
        return len(self.flows)

    def garbage_collect(self, latest_time) -> None:
        # TODO: Garbage Collection / Feature Extraction should have a separate thread
        self.logger.debug(f"Garbage Collection Began. Flows = {len(self.flows)}")
        keys = list(self.flows.keys())
        for k in keys:
            flow = self.flows.get(k)

            if (
                latest_time is not None
                and latest_time - flow.latest_timestamp < EXPIRED_UPDATE
                and flow.duration < 90
            ):
                continue

            del self.flows[k]
        self.logger.debug(f"Garbage Collection Finished. Flows = {len(self.flows)}")


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
