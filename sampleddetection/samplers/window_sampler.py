import logging
from math import ceil
from pathlib import Path
from random import shuffle

import numpy as np
from scapy.all import PcapReader, rdpcap
from scapy.plist import PacketList

from ..utils import setup_logger

# import PcaketList


class UniformWindowSampler:
    def __init__(
        self,
        path: str,
        window_skip: float = 1.0,
        window_length: float = 1.0,
        amount_windows: int = 100,
    ):
        """
        Sampler that will uniformly sample windows from the capture file.
        Args:
            path (str): Path to the .pcap file
            window_skip (float, optional): Time to skip between windows. Defaults to 1.0.
            window_length (float, optional): Length of each window. Defaults to 1.0.
            amount_windows (int, optional): Amount of windows to sample. Defaults to 100.
            locating_delta(float, optional): Delta used to determine if we have come close to a packet or not.
        """
        # Ensure path exists
        assert Path(path).exists(), "The provided path does not exists"
        self.logger = setup_logger("UniformSampler", logging.DEBUG)

        self.amount_windows = amount_windows
        self.window_skip = window_skip
        self.window_length = window_length

        # Locate and Read the file
        self.logger.info("⚠️ Loading the capture file. This will likely take a while")
        self.capture = rdpcap(str(path))
        self.first_ts = self.capture[0].time  # First Timestamp
        self.last_ts = self.capture[-1].time  # Last Timestamp
        self.logger.info("✅ Loaded the capture file.")

        # Start the sampling
        self.logger.info("Creating window samples")
        self.windows_list = self._create_window_samples()

    def _create_window_samples(self):
        """
        Assuming properly loaded capture file, will sample from it uniformly some windows
        Disclaimer:
            I will not sample *in array* because I assume the distribution of packets across time will be very uneven.
            Thus I sample in time.
        """
        # duration = self.last_ts - self.first_ts

        # Chose random times with this duration
        random_times = np.random.uniform(
            low=self.first_ts, high=self.last_ts, size=self.amount_windows
        )
        windows_list = []
        # Do Binary Search on the capture to find the initial point for each packet
        for rand_time in random_times:
            cur_idx = self._binary_search(self.capture, rand_time)
            end_time = rand_time + self.window_length

            cur_packet = self.capture[cur_idx]
            # For each of these initial samples get its window.
            window_packet_list = []
            while self.capture[cur_idx].time < end_time:
                # Append Packet
                assert hasattr(  # TODO: remove this if it never really gets called.
                    self.capture[cur_idx], "time"
                ), "Apparently packet doe snot have sniff-timestamp"

                if (
                    "TCP" not in self.capture[cur_idx]
                    and "UDP" not in self.capture[cur_idx]
                ):
                    continue
                window_packet_list.append(self.capture[cur_idx])

                cur_idx += 1
            # When Done capturing we add the packet list:
            windows_list.append(window_packet_list)

        # TODO: Ensure good balance between labels.
        return windows_list

    def _binary_search(self, capture: PacketList, target_time: float):
        """
        Given a capture and a target time, will return the index of the first packet
        that is after the target time
        """
        # Initialize variables
        low = 0
        high = len(capture)
        mid = 0

        # Mid should be above our target_time.

        # Do binary search
        while high > low:
            mid = ceil(
                (high + low) / 2
            )  # CHECK: It *should* be ceil. Check nonetheless
            if target_time > capture[mid].time:
                low = mid + 1
            else:
                high = mid

        # Return the inde
        return mid

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
