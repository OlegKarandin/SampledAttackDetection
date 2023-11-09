"""
Base class for parsing through pcap files.
Based on `./scripts/cc2018_dataset_parser.py`
"""

import hashlib
import logging
from pathlib import Path, PosixPath
from typing import Dict

import pyshark
from scapy.all import PcapReader
from scapy.packet import Packet

from ..utils import setup_logger


class PCapWindowItr:
    def __init__(self, path: Path, window_skip: float, window_length: float):
        """Parser for .pcap files

        Args:
            path (Path): Path to the .pcap file
        """
        self.path: Path = path
        # self.cap = pyshark.FileCapture(
        # self.path,
        # keep_packets=True,
        # )
        self.cap = PcapReader(str(path))  # type: ignore
        self.window_length = window_length
        self.cur_window = []
        self.cur_time = 0.0
        self.logger = setup_logger("PCapWindowItr", logging.INFO)

        # Calculate length (in seconds) of capture time(or not, it might take too memory)
        # self.length = float(self.cap[-1].frame_info.time_epoch)

    def __iter__(self):
        """Iterate over windows

        Yields:
            [type]: [description]
        """
        no_timestamps = 0
        total_packets_analyzed = 0

        for packet in self.cap:
            # This is for the sake of memory(if we dont do this then we have to save packets in memory)
            if self.cur_time == 0.0:
                self.cur_time = packet.time  # type: ignore
            # Get next packet
            assert hasattr(
                packet, "time"
            ), "Apparently packet doe snot have sniff-timestamp"
            if "TCP" not in packet and "UDP" not in packet:
                no_timestamps += 1
                continue

            # Get time of packet
            # packet_time = float(packet.frame_info.time_epoch)
            packet_time = packet.time  # type: ignore

            # Check if packet is in current window
            if packet_time < self.cur_time + self.window_length:
                self.cur_window.append(packet)
                total_packets_analyzed += 1
            else:
                # Return current window
                self.logger.debug(
                    f"Done collecting first window. We have a total of f{total_packets_analyzed} packets analyzed"
                )
                yield self.cur_window

                # Reset current window
                self.cur_window = [packet]

                total_packets_analyzed = 0

                # Update current time
                self.cur_time += self.window_length
        self.logger.info(f"Total amount of missed timestamps: {no_timestamps}")
        return

    def __len__(self):
        """Length of current widnow

        Returns:
            [type]: [description]
        """
        return len(self.cur_window)

    def __getitem__(self, idx: int):
        """Get packet by index

        Args:
            idx (int): index of packet

        Returns:
            [type]: [description]
        """
        return self.cap[idx]

    def __str__(self):
        """String representation of parser

        Returns:
            [type]: [description]
        """
        return f"{self.path.name} - {len(self.cap)} packets"

    def __repr__(self):
        """Representation of parser

        Returns:
            [type]: [description]
        """
        return f"Parser(path={self.path}, num_packets={self.num_packets})"

    @property
    def packets(self):
        return self.cap

    @property
    def num_packets(self):
        return self.num_packets

    @staticmethod
    def get_hash(packet):
        return hashlib.md5(str(packet).encode("utf-8")).hexdigest()

    @staticmethod
    def get_protocol(packet):
        return packet.highest_layer
