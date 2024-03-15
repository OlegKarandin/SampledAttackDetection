"""
Functions that are fed list of packets and derive 
statitics relevant to machine learning algorithms
"""
from typing import Dict, List, Tuple, Union

# from pyshark.packet.packet import Packet
from scapy.all import Packet
from scapy.plist import PacketList

# from cicflowmeter.features.context.packet_direction import PacketDirection
from sampleddetection.datastructures.context.packet_direction import PacketDirection
from sampleddetection.datastructures.context.packet_flow_key import get_packet_flow_key

# from cicflowmeter.features.context.packet_flow_key import get_packet_flow_key
from sampleddetection.datastructures.flow import Flow

EXPIRED_UPDATE = 40


# TODO: I anticipate choking due to memory for large enough windows.
# might want to offload some of the values to a stream
def get_flows(packets: Union[List[Packet], PacketList]) -> Dict[Tuple, Flow]:
    """
    Given a list of packets, return a list of statistics
    relevant to machine learning algorithms
    """
    flows = {}
    packets_count = 0

    for packet in packets:
        # Dont accept packets without these transport protocols(because stats depend on these)
        # Count disambiguates between UDP packet clusters that are too distant in time. Makes them into new flows
        count = 0
        if "TCP" not in packet and "UDP" not in packet:
            continue

        direction = PacketDirection.FORWARD
        # Creates a key in hashtable
        packet_flow_key = get_packet_flow_key(packet, direction)
        flow = flows.get((packet_flow_key, count))

        packets_count += 1

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = flows.get((packet_flow_key, count))

        if flow is None:
            # If no flow exists create a new flow
            direction = PacketDirection.FORWARD
            flow = Flow(packet, direction)
            packet_flow_key = get_packet_flow_key(packet, direction)
            flows[(packet_flow_key, count)] = flow

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            # If the packet exists in the flow but the packet is sent
            # after too much of a delay than it is a part of a new flow.
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                expired += EXPIRED_UPDATE
                flow = flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    flows[(packet_flow_key, count)] = flow
                    break
        flow.add_packet(packet, direction)

    # For each flow in the dictionary get their statistics via value.get_data()

    return flows
    # return list(flows.values())


# Just so I can separete flow genration from stat generation
def flow_to_stats(flows: Dict[Tuple, Flow]) -> Dict[Tuple, Dict]:
    statistics = {}
    for key, value in flows.items():
        statistics[key] = value.get_data()
    return statistics
