import json
import logging
from datetime import datetime
from math import ceil
from pathlib import Path
from random import shuffle
from time import time
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

from sampleddetection.datastructures.context.packet_flow_key import get_packet_flow_key
from sampleddetection.datastructures.flowsession import SampledFlowSession
from sampleddetection.readers.readers import AbstractReader, CaptureReader, CSVReader

from ..common_lingo import ATTACK_TO_STRING
from ..utils import deprecated, epoch_to_clean, setup_logger


class DynamicWindowSampler:
    """
    WARNING: THIS DOES NOT TAKE PCAP FILES
    """

    NUM_WINDOWS_PER_SAMPLE = 1

    def __init__(
        self, path: Union[Path, None] = None, csvrdr: Union[CSVReader, None] = None
    ):
        path_str = path.__str__()
        self.logger = setup_logger(__class__.__name__, logging.DEBUG)
        if csvrdr == None:
            if path != None:
                assert path.exists() and path_str.endswith(  # type: ignore
                    ".csv"
                ), f"Path {path} does not exist or is invalid"

                self.logger.info(f"Loading the capture file {path_str}")
                if not path.exists():
                    raise ValueError(f"Provided path {path} does not exist")

                self.logger.info(f"Loading the capture file {path}")
                self.csvrdr = CSVReader(path)
            else:
                raise ValueError("Path is None")
        else:
            self.csvrdr: CSVReader = csvrdr

        self.first_sniff_time = self.csvrdr.first_sniff_time
        self.last_sniff_time = self.csvrdr.last_sniff_time
        self.max_idx = len(self.csvrdr.csv_df) - 1

    def sample(
        self,
        initial_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
    ) -> SampledFlowSession:
        # ) -> pd.DataFrame:
        """
        Will return a `SampledFlowSession` for a specific time window.
        `SampledFlowSession.get_data()` will get you a dictionary of statistics for all flows in that window.

        Parameters
        ~~~~~~~~~~
            - initial_precise: Whether we shoudl start precisely at the provided time or at the closest packet to it
        """
        # packet_list = []

        idx_firstpack = binary_search(self.csvrdr, initial_time)
        if initial_precise:
            cur_time = (
                initial_time + window_skip
            )  # Assumes that we will start `window_skip` after inference
        else:
            cur_time = (
                self.csvrdr[idx_firstpack].time - 1e-7
            )  # Assuming micro sec packet capture

        next_stop = cur_time + window_length

        flow_session = SampledFlowSession(
            sampwindow_length=window_length, sample_initpos=initial_time
        )

        curpack = self.csvrdr[idx_firstpack]
        self.logger.debug(f"starting at time {cur_time}")

        idx_curpack = idx_firstpack
        for _ in range(self.NUM_WINDOWS_PER_SAMPLE):
            while (
                idx_curpack < self.max_idx and self.csvrdr[idx_curpack].time < next_stop
            ):
                curpack = self.csvrdr[idx_curpack]
                self.logger.debug(
                    f"at idx {idx_curpack} we see a timestamp of {curpack.time}({epoch_to_clean(curpack.time)})"
                )
                # self.logger.debug(f"cur_time {cur_time} stopping at {next_stop}")
                # self.logger.debug(
                #     f"Packet at time {cur_time} has label {ATTACK_TO_STRING[curpack.label]}"
                # )
                flow_session.on_packet_received(curpack)
                idx_curpack += 1

            # TODO: Create a check to ensure we are not going over the margin here
            cur_time = next_stop + window_skip
            next_stop = cur_time + window_length

        # self.logger.debug(
        #     f"Size of these flows is {str([len(f) for f in flow_session.flows.values()])}. "
        # )
        return flow_session

    @deprecated(
        reason="We are not using window_length as a class property anymore",
        date="Mar 15, 2024",
    )
    def sample_n_flows(
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
            cur_packet = self.csvrdr[idx_curpack]
            cur_time = cur_packet.time

            flow_session.on_packet_received(cur_packet, cur_left_limit, cur_right_limit)

            cur_num_flows = flow_session.num_flows()
            idx_curpack += 1
            # TODO: check if we hit the limits of the pcap file. If so we may want to start again

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


def binary_search(cpt_rdr: AbstractReader, target_time: float) -> int:
    """
    Given a capture and a target time, will return the index of the first packet
    that is after the target time
    """
    # Initialize variables
    # self.logger.debug(f"The initial high is  {high}")
    low, high = binary_till_two(target_time, cpt_rdr)
    # Once we have two closest elements we check the closes
    # Argmax it
    if target_time <= float(cpt_rdr[low].time) and abs(
        target_time - float(cpt_rdr[low].time)
    ) < abs(target_time - float(cpt_rdr[high].time)):
        return low
    else:
        return high


def binary_search_upper(cpt_rdr: AbstractReader, target_time: float) -> int:
    _, high = binary_till_two(target_time, cpt_rdr)
    return high


def binary_till_two(target_time: float, pckt_rdr: AbstractReader) -> Tuple[int, int]:
    """
    Will do binary search until only two elements remain
    """
    low: int = 0
    high: int = len(pckt_rdr) - 1
    mid: int = 0
    # Do binary search
    # while high > low:
    while (high - low) != 1:
        mid = ceil((high + low) / 2)  # CHECK: It *should* be ceil. Check nonetheless
        if target_time > pckt_rdr.getTimestamp(mid):
            low = mid
        else:
            high = mid
    return low, high
