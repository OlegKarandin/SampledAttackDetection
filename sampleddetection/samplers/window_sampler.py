import json
import logging
from datetime import datetime
from math import ceil
from pathlib import Path
from random import shuffle
from time import time
from typing import Tuple, Union

import numpy as np
import torch
from scapy.all import Packet, PcapReader, rdpcap, wrpcap
from scapy.plist import PacketList
from tqdm import tqdm

from sampleddetection.common_lingo import RelevantStats
from sampleddetection.datastructures.context.packet_flow_key import get_packet_flow_key
from sampleddetection.datastructures.flowsession import SampledFlowSession
from sampleddetection.readers.readers import (
    CaptureReader,
    CSVReader,
    PartitionCaptureReader,
)

from ..utils import setup_logger

# import PcaketList
MAX_MEM = 15e9  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 250e6  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 100e6  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 2 ** (26.575)
# MAX_MEM = 2 ** (23.25)


class DynamicWindowSampler:
    def __init__(self, path: str):
        assert Path(path).exists()
        self.logger = setup_logger(__class__.__name__, logging.DEBUG)

        # To be filled later
        self.window_skip = 0.0
        self.window_length = 0.0

        self.logger.info(f"Loading the capture file {path}")
        self.caprdr = CSVReader(Path(path))

    def sample_w_freq_n_win(
        self, initial_time: float, min_num_flows: int
    ) -> SampledFlowSession:
        # OPTIM: so much to do here. We might need to depend on a c++ library to do this.
        """
        Will create `samples` samples from which to form current statistics
        """

        idx_curpack = binary_search(self.caprdr, initial_time)
        cur_num_flows = 0

        cur_time = initial_time
        cur_left_limit = initial_time
        cur_right_limit = initial_time + self.window_length

        # Create New Flow Session
        flow_session = SampledFlowSession()

        while cur_num_flows < min_num_flows:
            # Keep going through packets
            cur_packet = self.caprdr[idx_curpack]
            cur_time = cur_packet.time

            flow_session.on_packet_received(cur_packet, cur_left_limit, cur_right_limit)

            cur_num_flows = flow_session.num_flows()
            idx_curpack += 1
            # TODO: check if we hit the limits of the pcap file. If so we may want to start again

            # Entrando aca que pedos
            if cur_time > cur_right_limit:
                cur_left_limit = cur_right_limit + self.window_skip
                cur_right_limit = cur_left_limit + self.window_length

        return flow_session


class UniformWindowSampler:
    def __init__(
        self,
        path: str,
        window_skip: float = 1.0,
        window_length: float = 1.0,
        amount_windows: int = 100,
        partition_loc: str = "./partitions",
    ):
        """
        Sampler will uniformly sample windows from the capture file.
        Args:
            path (str): Path to the .pcap file
            window_skip (float, optional): Time to skip between windows. Defaults to 1.0.
            window_length (float, optional): Length of each window. Defaults to 1.0.
            amount_windows (int, optional): Amount of windows to sample. Defaults to 100.
            locating_delta(float, optional): Delta used to determine if we have come close to a packet or not.
            partition_loc (str, optional): Location of the partitions. Defaults to "./partitions".
        """
        # Ensure path exists
        assert Path(path).exists(), "The provided path does not exists"
        self.logger = setup_logger("UniformSampler", logging.DEBUG)

        self.amount_windows = amount_windows
        self.window_skip = window_skip
        self.window_length = window_length

        # Structure doing most of the heavy lifting
        self.caprdr = CaptureReader(Path(path))
        # self.caprdr.partition()

        # Locate and Read the file
        self.logger.info("⚠️ Loading the capture file. This will likely take a while")
        self.first_ts = self.caprdr.first_sniff_time
        self.last_ts = self.caprdr.last_sniff_time
        self.logger.info("✅ Loaded the capture file.")

        # Start the sampling
        self.logger.info("Creating window samples")
        self.windows_list = self._create_window_samples()

    def set_new_sampling_params(self, window_skip, window_length):
        self.window_skip = window_skip
        self.window_length = window_length

    def _create_window_samples(self):
        """
        Assuming properly loaded capture file, will uniformly sample from it some windows
        Disclaimer:
            I will not sample *in array* because I assume the distribution of packets across time will be very uneven.
            Thus I sample in time.
        """
        # duration = self.last_ts - self.first_ts

        # Chose random times with this duration
        np.random.seed(42)
        windows_list = []
        # Do Binary Search on the capture to find the initial point for each packet

        times_bar = tqdm(
            total=self.amount_windows,
            desc="Looking through sampling windows",
            leave=True,
            position=0,
        )

        while len(windows_list) < self.amount_windows:
            win_start_time = np.random.uniform(
                low=self.first_ts, high=self.last_ts, size=1
            )[0]
            cur_idx = binary_search(self.caprdr, win_start_time)
            win_end_time = win_start_time + self.window_length

            # cur_packet = self.capture[cur_idx]
            # For each of these initial samples get its window.
            window_packet_list = []
            adding_bar = tqdm(desc="Adding packets", leave=True, position=1)

            while self.caprdr[cur_idx].time < win_end_time:
                # Append Packet
                assert hasattr(  # TOREM:
                    self.caprdr[cur_idx], "time"
                ), "Apparently packet does not have sniff-timestamp"

                adding_bar.update(1)
                if (
                    "TCP" not in self.caprdr[cur_idx]
                    and "UDP" not in self.caprdr[cur_idx]
                ):
                    cur_idx += 1
                    continue
                window_packet_list.append(self.caprdr[cur_idx])

                cur_idx += 1
            # When done capturing we add the packet list:
            if len(window_packet_list) > 0:
                dt_start = datetime.fromtimestamp(int(win_start_time))
                dt_end = datetime.fromtimestamp(int(win_end_time))
                self.logger.debug(
                    f"Adding a new window of packets between {dt_start} and {dt_end}"
                )
                for cap in window_packet_list[-1]:
                    dt_time = datetime.fromtimestamp(int(cap.time))
                    self.logger.debug(f"\t time:{dt_time} summary:{cap.summary}")
                windows_list.append(window_packet_list)

                times_bar.update(1)

        # TODO: Ensure good balance between labels.
        return windows_list

    def get_first_packet_in_window(self, init_pos, win_length):
        """
        Will look forward in time for win_length units of time from init_pos.
        Will then take the first packet it finds
        TODO: maybe implement this if we find it necessary
        """
        pass

    def uniform_window_sample():
        """
        Assuming the list is ready and balance it will just iterate over it.
        """

        pass

    def __iter__(self):
        # Scramble window_list
        shuffle(self.windows_list)
        for window in self.windows_list:
            yield window


def binary_search(cpt_rdr: CaptureReader, target_time: float):
    """
    Given a capture and a target time, will return the index of the first packet
    that is after the target time
    """
    # Initialize variables
    # self.logger.debug(f"The initial high is  {high}")
    low, high = binary_till_two(target_time, cpt_rdr)
    # Once we have two closest elements we check the closes
    # Argmax it
    if abs(target_time - float(cpt_rdr[low].time)) < abs(
        target_time - float(cpt_rdr[high].time)
    ):
        return low
    else:
        return high


def binary_till_two(target_time: float, cpt_rdr: CaptureReader):
    """
    Will do binary search until only two elements remain
    """
    low: int = 0
    high: int = len(cpt_rdr)
    mid: int = 0
    # Do binary search
    # while high > low:
    while (high - low) != 1:
        mid = ceil((high + low) / 2)  # CHECK: It *should* be ceil. Check nonetheless
        if target_time > float(cpt_rdr[mid].time):
            low = mid
        else:
            high = mid
    return low, high
