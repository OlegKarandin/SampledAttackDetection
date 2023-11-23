"""
@author: Oleg Karandin
more info: ?
"""
import hashlib

import numpy as np
import pandas as pd
import pyshark

ETHERNET_HEX_TO_STRING = {
    "0x00000800": "IPv4",
    "0x0800": "IPv4",
    "0x86dd": "IPv6",
    "0x000086dd": "IPv6",
    "0x0806": "ARP",
    "0x00000806": "ARP",
    "0x8035": "RARP",
    "0x8100": "IPX",
    "0x8137": "IPX",
}
PROTOCOL_DICT = {
    "1": "ICMP",
    "2": "IGMP",
    "6": "TCP",
    "17": "UDP",
    "41": "IPv6",
    "89": "OSPF",
    "132": "SCTP",
}

PROTOCOL_IPV6_DICT = {
    "6": "TCP",
    "17": "UDP",
    "58": "ICMPv6",
    "89": "OSPFv3",
}
FLOWS = {}

NUM_PACKETS = 100
TOTAL_NUM_PACKETS = 13788878
if __name__ == "__main__":
    # the dataste with flow features and labels
    cicids2017_df = pd.read_csv("./Wednesday-workingHours.pcap_ISCX.csv")
    cicids2017_df.drop_duplicates(inplace=True)

    # the dataset with captured packets
    cap = pyshark.FileCapture("./Wednesday-WorkingHours.pcap")

    # read the next packet
    for idx, packet in enumerate(cap):
        print(packet)
        print(packet.layers)
        print(packet.frame_info)
        print(packet.frame_info.time)
        print(packet.frame_info.time_epoch)

        if idx % 10000 == 0:
            progress = idx / TOTAL_NUM_PACKETS * 100
            print(f"PACKET {idx}/{TOTAL_NUM_PACKETS}. {progress} % ")

        # parse the packet
        packet_ethernet_type = None
        packet_protocol_name = None
        packet_protocol_number = None
        packet_df_flag = None
        packet_mf_flag = None
        packet_ipv4_size = 0
        packet_dst_addr = None
        packet_src_addr = None
        packet_src_port = None
        packet_dst_port = None
        packet_TCP_FIN_flag = None
        packet_TCP_SYN_flag = None
        packet_TCP_RST_flag = None
        packet_TCP_PSH_flag = None
        packet_TCP_ACK_flag = None
        packet_TCP_URG_flag = None

        #####ETHERNET LAYER
        try:
            ethernet_layer = packet.eth
            # print(f"Ethernet Layer: {ethernet_layer}")
            packet_ethernet_type = ETHERNET_HEX_TO_STRING[ethernet_layer.type]
        except Exception as e:
            print(f"Error in Ethernet Layer: {e}")
            # print(packet)
            # break

        ######IP LAYER
        if packet_ethernet_type in ["IPv4"]:
            try:
                ip_layer = packet.ip
                # print(packet)
                if ip_layer:
                    # print(f"IP Layer: {ip_layer}")
                    packet_protocol_name = PROTOCOL_DICT[ip_layer.proto]
                    packet_protocol_number = ip_layer.proto
                    packet_df_flag = ip_layer.flags_df
                    packet_mf_flag = ip_layer.flags_mf
                    packet_ipv4_size = ip_layer.len
                    packet_dst_addr = ip_layer.dst
                    packet_src_addr = ip_layer.src
            except Exception as e:
                print(f"Error in IPv4 Layer: {e}")
                # print(packet)
                # break

        elif packet_ethernet_type in ["IPv6"]:
            try:
                ip_layer = packet.ipv6

                if ip_layer:
                    # print(f"IP Layer: {ip_layer}")
                    # print(ip_layer.nxt)
                    packet_protocol_name = PROTOCOL_IPV6_DICT[ip_layer.nxt]
                    packet_protocol_number = ip_layer.nxt
                    # packet_ipv4_size 	= int(ip_layer.plen) + 40
                    # packet_dst_addr 	= ip_layer.dst
                    # packet_src_addr 	= ip_layer.src
                else:
                    print("Warning: IPv6 layer not found")

            except Exception as e:
                print(f"Error in IPv6 Layer: {e}")
                # print(packet)
                # break

        #####TCP LAYER
        if packet_protocol_name == "TCP":
            try:
                tcp_layer = packet.tcp
                packet_src_port = int(tcp_layer.srcport)
                packet_dst_port = int(tcp_layer.dstport)

                tcp_flags = int(
                    tcp_layer.flags, 16
                )  # Convert the hex string to an integer
                packet_TCP_FIN_flag = (
                    1 if (tcp_flags & 1) != 0 else 0
                )  # Extract the FIN bit
                packet_TCP_SYN_flag = (
                    1 if (tcp_flags & 2) != 0 else 0
                )  # Extract the SYN bit
                packet_TCP_RST_flag = (
                    1 if (tcp_flags & 4) != 0 else 0
                )  # Extract the RST bit
                packet_TCP_PSH_flag = (
                    1 if (tcp_flags & 8) != 0 else 0
                )  # Extract the PSH bit
                packet_TCP_ACK_flag = (
                    1 if (tcp_flags & 16) != 0 else 0
                )  # Extract the ACK bit
                packet_TCP_URG_flag = (
                    1 if (tcp_flags & 32) != 0 else 0
                )  # Extract the URG bit

            except Exception as e:
                print(f"Error in TCP Layer: {e}")
                # break
        else:
            # print("NO TCP")
            pass

        # UDP LAYER
        if packet_protocol_name == "UDP":
            try:
                udp_layer = packet.udp
                packet_src_port = udp_layer.srcport
                packet_dst_port = udp_layer.dstport
            except Exception as e:
                print(f"Error in UDP Layer: {e}")
                # break

        else:
            pass
            # print("NO UDP")

        # some parts of flow id were not parsed correctly
        if (
            packet_src_addr == None
            or packet_dst_addr == None
            or packet_src_port == None
            or packet_dst_port == None
            or packet_protocol_number == None
        ):
            continue

        else:
            flow_str = "{}-{}-{}-{}-{}".format(
                packet_src_addr,
                packet_dst_addr,
                packet_src_port,
                packet_dst_port,
                packet_protocol_number,
            )

            # look for flows with this ID
            label_array = cicids2017_df[cicids2017_df["Flow ID"] == flow_str][
                " Label"
            ].to_numpy()
            label_values, _ = np.unique(label_array, return_counts=False)

            # TODO: compare packet.frame_info.time with the cicids2017_df[' Timestamp']

            # more than 1 flow fits the ID
            if label_values.size > 1:
                print(flow_str, label_array, label_values, label_values.size)

            if flow_str in FLOWS:
                FLOWS[flow_str]["Packet Count"] += 1

            else:
                FLOWS[flow_str] = {"Packet Count": 1}

        # Convert the string to UTF-8 and apply the SHA-256 hash function
        """flow_hash = hashlib.sha256(flow_str.encode('utf-8')).hexdigest()
		print(f"Flow hash: {flow_hash}")"""

        # FEATURES
        """print("PACKET FEATURES:")

		print(f"Packet Ethernet Type: {packet_ethernet_type}")
		print(f"Packet Protocol: {packet_protocol_name}")

		print(f"Packet Source Address: {packet_src_addr}")
		print(f"Packet Destination Address: {packet_dst_addr}")

		print(f"Packet Source Port: {packet_src_port}")
		print(f"Packet Destination port: {packet_dst_port}")"""

        if idx >= NUM_PACKETS:
            break

    cap.close()
