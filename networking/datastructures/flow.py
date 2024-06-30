from enum import Enum
from typing import Any, NamedTuple

from scapy.all import Packet

import pandas as pd
import json
from networking.common_lingo import ATTACK_TO_STRING, Attack
from networking.datastructures.packet_like import PacketLike
from sampleddetection.common_lingo import TimeWindow
from sampleddetection.utils import get_statistics, setup_logger

from . import constants
from .context import packet_flow_key
from .context.packet_direction import PacketDirection
from .flag_count import FlagCount
from .flow_bytes import FlowBytes
from .packet_count import PacketCount
from .packet_length import PacketLength
from .packet_time import PacketTime


class Flow:
    """This class summarizes the values of the features of the network flows"""

    def __init__(self, packet: Any, direction: PacketDirection):
        """This method initializes an object from the Flow class.

        Args:
            packet (Any): A packet from the network.
            direction (Enum): The direction the packet is going ove the wire.
        """
        (
            self.dest_ip,
            self.src_ip,
            self.src_port,
            self.dest_port,
        ) = packet_flow_key.get_packet_flow_key(packet, direction)

        # DEBUG:remove this logger pls
        # self.logger = setup_logger(__class__.__name__, overwrite=False)

        self.packets = []
        self.flow_interarrival_time = []
        self.latest_timestamp = 0
        self.start_timestamp = 0
        self.init_window_size = {
            PacketDirection.FORWARD: 0,
            PacketDirection.REVERSE: 0,
        }

        self.time_window = TimeWindow(-1, -1)

        self.start_active = 0
        self.last_active = 0
        self.active = []
        self.idle = []

        self.forward_bulk_last_timestamp = 0
        self.forward_bulk_start_tmp = 0
        self.forward_bulk_count = 0
        self.forward_bulk_count_tmp = 0
        self.forward_bulk_duration = 0
        self.forward_bulk_packet_count = 0
        self.forward_bulk_size = 0
        self.forward_bulk_size_tmp = 0
        self.backward_bulk_last_timestamp = 0
        self.backward_bulk_start_tmp = 0
        self.backward_bulk_count = 0
        self.backward_bulk_count_tmp = 0
        self.backward_bulk_duration = 0
        self.backward_bulk_packet_count = 0
        self.backward_bulk_size = 0
        self.backward_bulk_size_tmp = 0

        # Benign or type of malign
        self.label: Attack = Attack.BENIGN

    def get_data(self) -> dict:
        """This method obtains the values of the features extracted from each flow.

        Note:
            Only some of the network data plays well together in this list.
            Time-to-live values, window values, and flags cause the data to
            separate out too much.

        Returns:
           list: returns a List of values to be outputted into a csv file.

        """

        flow_bytes = FlowBytes(self)
        flag_count = FlagCount(self.packets)
        packet_count = PacketCount(self)
        packet_length = PacketLength(self)
        packet_time = PacketTime(self)
        flow_iat = get_statistics(self.flow_interarrival_time)
        forward_iat = get_statistics(
            packet_time.get_packet_iat(PacketDirection.FORWARD)
        )
        backward_iat = get_statistics(
            packet_time.get_packet_iat(PacketDirection.REVERSE)
        )
        active_stat = get_statistics(self.active)
        idle_stat = get_statistics(self.idle)

        data = {
            # Some Debugging information
            "start_ts": packet_time.first_ts,
            "start_timestamp": packet_time.to_str(packet_time.first_ts),
            "end_timestamp": packet_time.to_str(packet_time.last_ts),
            # Basic IP information
            "src_ip": self.src_ip,
            "dst_ip": self.dest_ip,
            "src_port": self.src_port,
            "dst_port": self.dest_port,
            "protocol": self.protocol,
            # Basic information from packet times
            "timestamp": packet_time.get_time_stamp(),
            "flow_duration": packet_time.get_duration(),
            "flow_byts_s": flow_bytes.get_rate(),
            "flow_pkts_s": packet_count.get_rate(),
            "fwd_pkts_s": packet_count.get_rate(PacketDirection.FORWARD),
            "bwd_pkts_s": packet_count.get_rate(PacketDirection.REVERSE),
            # Count total packets by direction
            "tot_fwd_pkts": packet_count.get_total(PacketDirection.FORWARD),
            "tot_bwd_pkts": packet_count.get_total(PacketDirection.REVERSE),
            # Statistical info obtained from Packet lengths
            "totlen_fwd_pkts": packet_length.get_total(PacketDirection.FORWARD),
            "totlen_bwd_pkts": packet_length.get_total(PacketDirection.REVERSE),
            "fwd_pkt_len_max": float(packet_length.get_max(PacketDirection.FORWARD)),
            "fwd_pkt_len_min": float(packet_length.get_min(PacketDirection.FORWARD)),
            "fwd_pkt_len_mean": float(packet_length.get_mean(PacketDirection.FORWARD)),
            "fwd_pkt_len_std": float(packet_length.get_std(PacketDirection.FORWARD)),
            "bwd_pkt_len_max": float(packet_length.get_max(PacketDirection.REVERSE)),
            "bwd_pkt_len_min": float(packet_length.get_min(PacketDirection.REVERSE)),
            "bwd_pkt_len_mean": float(packet_length.get_mean(PacketDirection.REVERSE)),
            "bwd_pkt_len_std": float(packet_length.get_std(PacketDirection.REVERSE)),
            "pkt_len_max": packet_length.get_max(),
            "pkt_len_min": packet_length.get_min(),
            "pkt_len_mean": float(packet_length.get_mean()),
            "pkt_len_std": float(packet_length.get_std()),
            "pkt_len_var": float(packet_length.get_var()),
            "fwd_header_len": flow_bytes.get_forward_header_bytes(),
            "bwd_header_len": flow_bytes.get_reverse_header_bytes(),
            "fwd_seg_size_min": flow_bytes.get_min_forward_header_bytes(),
            "fwd_act_data_pkts": packet_count.has_payload(PacketDirection.FORWARD),
            # Flows Interarrival Time
            "flow_iat_mean": float(flow_iat["mean"]),
            "flow_iat_max": float(flow_iat["max"]),
            "flow_iat_min": float(flow_iat["min"]),
            "flow_iat_std": float(flow_iat["std"]),
            "fwd_iat_tot": forward_iat["total"],
            "fwd_iat_max": float(forward_iat["max"]),
            "fwd_iat_min": float(forward_iat["min"]),
            "fwd_iat_mean": float(forward_iat["mean"]),
            "fwd_iat_std": float(forward_iat["std"]),
            "bwd_iat_tot": float(backward_iat["total"]),
            "bwd_iat_max": float(backward_iat["max"]),
            "bwd_iat_min": float(backward_iat["min"]),
            "bwd_iat_mean": float(backward_iat["mean"]),
            "bwd_iat_std": float(backward_iat["std"]),
            # Flags statistics
            "fwd_psh_flags": flag_count.has_flag("PSH", PacketDirection.FORWARD),
            "bwd_psh_flags": flag_count.has_flag("PSH", PacketDirection.REVERSE),
            "fwd_urg_flags": flag_count.has_flag("URG", PacketDirection.FORWARD),
            "bwd_urg_flags": flag_count.has_flag("URG", PacketDirection.REVERSE),
            "fin_has_flag": flag_count.has_flag("FIN"),
            "syn_has_flag": flag_count.has_flag("SYN"),
            "rst_has_flag": flag_count.has_flag("RST"),
            "psh_has_flag": flag_count.has_flag("PSH"),
            "ack_has_flag": flag_count.has_flag("ACK"),
            "urg_has_flag": flag_count.has_flag("URG"),
            "ece_has_flag": flag_count.has_flag("ECE"),
            "cwr_has_flag": flag_count.has_flag("CWR"),
            # Actual count
            "fin_flag_cnt": flag_count.count("FIN"),
            "syn_flag_cnt": flag_count.count("SYN"),
            "rst_flag_cnt": flag_count.count("RST"),
            "psh_flag_cnt": flag_count.count("PSH"),
            "ack_flag_cnt": flag_count.count("ACK"),
            "urg_flag_cnt": flag_count.count("URG"),
            "ece_flag_cnt": flag_count.count("ECE"),
            "cwr_flag_cnt": flag_count.count("CWR"),
            # Response Time
            "down_up_ratio": packet_count.get_down_up_ratio(),
            "pkt_size_avg": packet_length.get_avg(),
            "init_fwd_win_byts": self.init_window_size[PacketDirection.FORWARD],
            "init_bwd_win_byts": self.init_window_size[PacketDirection.REVERSE],
            "active_max": float(active_stat["max"]),
            "active_min": float(active_stat["min"]),
            "active_mean": float(active_stat["mean"]),
            "active_std": float(active_stat["std"]),
            "idle_max": float(idle_stat["max"]),
            "idle_min": float(idle_stat["min"]),
            "idle_mean": float(idle_stat["mean"]),
            "idle_std": float(idle_stat["std"]),
            "fwd_byts_b_avg": float(
                flow_bytes.get_bytes_per_bulk(PacketDirection.FORWARD)
            ),
            "fwd_pkts_b_avg": float(
                flow_bytes.get_packets_per_bulk(PacketDirection.FORWARD)
            ),
            "bwd_byts_b_avg": float(
                flow_bytes.get_bytes_per_bulk(PacketDirection.REVERSE)
            ),
            "bwd_pkts_b_avg": float(
                flow_bytes.get_packets_per_bulk(PacketDirection.REVERSE)
            ),
            "fwd_blk_rate_avg": float(
                flow_bytes.get_bulk_rate(PacketDirection.FORWARD)
            ),
            "bwd_blk_rate_avg": float(
                flow_bytes.get_bulk_rate(PacketDirection.REVERSE)
            ),
        }

        # Duplicated features
        data["fwd_seg_size_avg"] = data["fwd_pkt_len_mean"]
        data["bwd_seg_size_avg"] = data["bwd_pkt_len_mean"]
        data["cwe_flag_count"] = data["fwd_urg_flags"]
        data["subflow_fwd_pkts"] = data["tot_fwd_pkts"]
        data["subflow_bwd_pkts"] = data["tot_bwd_pkts"]
        data["subflow_fwd_byts"] = data["totlen_fwd_pkts"]
        data["subflow_bwd_byts"] = data["totlen_bwd_pkts"]

        # CHECK: for critical errros
        if any(
            [
                data["fwd_iat_min"] < 0,
                data["pkt_len_min"] < 0,
            ]
        ):
            # DUMP all the packets into a files
            print("ERROR: Incorrect values for statistics. Will share them in a dump.")
            with open("dump.csv", "w") as f:
                cols = self.packets[0][0].row.index.values
                f.write(",".join(cols) + "\n")
                for packet, _ in self.packets:
                    # pack_info = packet.row.apply({lambda x: str(x)}).values
                    pack_info = [str(e) for e in packet.row.values]
                    f.write(",".join(pack_info) + "\n")

            # Let me alsdo dump all attributes to see which ones are negative
            with open("dump.json", "w") as f:
                # Convert all of data into pretty json
                json_data = json.dumps(data, indent=3)
                f.write(json_data)
            exit()

        # Finally the label
        data["label"] = ATTACK_TO_STRING[self.label]

        return data

    def add_packet(self, packet: PacketLike, direction: PacketDirection) -> None:
        """Adds a packet to the current list of packets.

        Args:
            packet: Packet to be added to a flow
            direction: The direction the packet is going in that flow

        """
        # Add Label
        if packet.label != Attack.BENIGN:
            if self.label == Attack.BENIGN:
                self.label = packet.label
            elif self.label != Attack.BENIGN and packet.label != self.label:
                # TODO: For Sanity Check only, streamline once sure
                raise ValueError(
                    f"We have flow already labeled as {self.label} and new packet labeled as {packet.label}"
                )

        # Normal Operations
        self.time_window.start = min(self.time_window.start, packet.time)
        self.time_window.end = max(self.time_window.end, packet.time)

        self.packets.append((packet, direction))

        self.update_flow_bulk(packet, direction)
        self.update_subflow(packet)

        if self.start_timestamp != 0:
            self.flow_interarrival_time.append((packet.time - self.latest_timestamp))
            # self.logger.debug(
            #     f"Calculation of flow_interarrival_time {self.flow_interarrival_time[-1]}\n\t"
            #     f"With current packet time = {packet.time}, self.latest_timestamp: {self.latest_timestamp}"
            # )

        self.latest_timestamp = max([packet.time, self.latest_timestamp])

        if "TCP" in packet:
            if (
                direction == PacketDirection.FORWARD
                and self.init_window_size[direction] == 0
            ):
                self.init_window_size[direction] = packet.tcp_window
            elif direction == PacketDirection.REVERSE:
                self.init_window_size[direction] = packet.tcp_window

        # First packet of the flow
        if self.start_timestamp == 0:
            self.start_timestamp = packet.time
            # self.logger.debug(f"Adding first time_stamp {packet.time}")
            self.protocol = packet.layers[-1]

    def update_subflow(self, packet):
        """Update subflow

        Args:
            packet: Packet to be parse as subflow

        """
        last_timestamp = (
            self.latest_timestamp if self.latest_timestamp != 0 else packet.time
        )
        if (packet.time - last_timestamp) > constants.CLUMP_TIMEOUT:
            self.update_active_idle(packet.time - last_timestamp)

    def update_active_idle(self, current_time):
        """Adds a packet to the current list of packets.

        Args:
            packet: Packet to be update active time

        """
        if (current_time - self.last_active) > constants.ACTIVE_TIMEOUT:
            # Creates a new subflow
            duration = abs(float(self.last_active - self.start_active))
            if duration > 0:
                self.active.append(duration)
            self.idle.append((current_time - self.last_active))
            self.start_active = current_time
            self.last_active = current_time
        else:
            self.last_active = current_time

    def update_flow_bulk(self, packet: PacketLike, direction):
        """Update bulk flow

        Args:
            packet: Packet to be parse as bulk

        """
        # payload_size = len(PacketCount.get_payload(packet))
        payload_size = packet.payload_size
        if payload_size == 0:
            return
        if direction == PacketDirection.FORWARD:
            if self.backward_bulk_last_timestamp > self.forward_bulk_start_tmp:
                self.forward_bulk_start_tmp = 0
            if self.forward_bulk_start_tmp == 0:
                self.forward_bulk_start_tmp = packet.time
                self.forward_bulk_last_timestamp = packet.time
                self.forward_bulk_count_tmp = 1
                self.forward_bulk_size_tmp = payload_size
            else:
                if (
                    packet.time - self.forward_bulk_last_timestamp
                ) > constants.CLUMP_TIMEOUT:
                    self.forward_bulk_start_tmp = packet.time
                    self.forward_bulk_last_timestamp = packet.time
                    self.forward_bulk_count_tmp = 1
                    self.forward_bulk_size_tmp = payload_size
                else:  # Add to bulk
                    self.forward_bulk_count_tmp += 1
                    self.forward_bulk_size_tmp += payload_size
                    if self.forward_bulk_count_tmp == constants.BULK_BOUND:
                        self.forward_bulk_count += 1
                        self.forward_bulk_packet_count += self.forward_bulk_count_tmp
                        self.forward_bulk_size += self.forward_bulk_size_tmp
                        self.forward_bulk_duration += (
                            packet.time - self.forward_bulk_start_tmp
                        )
                    elif self.forward_bulk_count_tmp > constants.BULK_BOUND:
                        self.forward_bulk_packet_count += 1
                        self.forward_bulk_size += payload_size
                        self.forward_bulk_duration += (
                            packet.time - self.forward_bulk_last_timestamp
                        )
                    self.forward_bulk_last_timestamp = packet.time
        else:
            if self.forward_bulk_last_timestamp > self.backward_bulk_start_tmp:
                self.backward_bulk_start_tmp = 0
            if self.backward_bulk_start_tmp == 0:
                self.backward_bulk_start_tmp = packet.time
                self.backward_bulk_last_timestamp = packet.time
                self.backward_bulk_count_tmp = 1
                self.backward_bulk_size_tmp = payload_size
            else:
                if (
                    packet.time - self.backward_bulk_last_timestamp
                ) > constants.CLUMP_TIMEOUT:
                    self.backward_bulk_start_tmp = packet.time
                    self.backward_bulk_last_timestamp = packet.time
                    self.backward_bulk_count_tmp = 1
                    self.backward_bulk_size_tmp = payload_size
                else:  # Add to bulk
                    self.backward_bulk_count_tmp += 1
                    self.backward_bulk_size_tmp += payload_size
                    if self.backward_bulk_count_tmp == constants.BULK_BOUND:
                        self.backward_bulk_count += 1
                        self.backward_bulk_packet_count += self.backward_bulk_count_tmp
                        self.backward_bulk_size += self.backward_bulk_size_tmp
                        self.backward_bulk_duration += (
                            packet.time - self.backward_bulk_start_tmp
                        )
                    elif self.backward_bulk_count_tmp > constants.BULK_BOUND:
                        self.backward_bulk_packet_count += 1
                        self.backward_bulk_size += payload_size
                        self.backward_bulk_duration += (
                            packet.time - self.backward_bulk_last_timestamp
                        )
                    self.backward_bulk_last_timestamp = packet.time

    @property
    def duration(self):
        return self.latest_timestamp - self.start_timestamp

    def __len__(self):
        return len(self.packets)
