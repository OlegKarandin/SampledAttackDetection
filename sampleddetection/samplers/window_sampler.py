import json
import logging
from datetime import datetime
from math import ceil
from pathlib import Path
from random import shuffle
from typing import Tuple, Union

import numpy as np
import torch
from scapy.all import Packet, PcapReader, rdpcap, wrpcap
from scapy.plist import PacketList
from tqdm import tqdm

from sampleddetection.common_lingo import RelevantStats
from sampleddetection.datastructures.context.packet_flow_key import get_packet_flow_key
from sampleddetection.datastructures.flowsession import SampledFlowSession

from ..utils import setup_logger

# import PcaketList
MAX_MEM = 15e9  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 250e6  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 100e6  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 2 ** (26.575)
# MAX_MEM = 2 ** (23.25)


class CaptureReader:
    def __init__(self, scr_path: Path):
        assert str(scr_path).endswith(
            ".pcap"
        ), f"Provided path to capture is not a pcap file"
        self.logger = setup_logger(
            "CaptureReader",
        )
        self.lf_location = src_path
        self.cur_cptfile_ptr = PacketList()
        self.packet_list = []
        self.last_sniff_time = 0.0
        self.first_sniff_time = 0.0
        self.length = 0

    def _load_data(
        self,
    ):
        pcap_rdr = PcapReader(str(self.lf_location))
        size_of_file = 0
        # CHECK: Json is not letting me dump `Decimal` types which
        #   packet.time is returning. Im afraid this might get in the way of precision.
        #   To get this draft finished Ill let it go for now but be careful
        #   Ill mark the two lines of relevant with "DTF"(decimal-to-float)

        packet = pcap_rdr.read_packet()
        self.first_sniff_time = float(packet.time)  # CHECK: DTF
        self.logger.info("Attempting to load monolithic file")
        last_packet = packet

        t_bar = tqdm(desc="Loading Monolithic File (GB)", leave=True, position=0)
        while packet != None:
            self.packet_list.append(packet)
            size_of_file += len(packet)
            t_bar.update(len(packet) / 10e9)
            last_packet = packet
            self.length += 1
            try:
                packet = pcap_rdr.read_packet()
            except EOFError:  # EOF
                break

        self.last_sniff_time = float(last_packet.time)
        self.logger.info(f"Loaded the monolithic file with size {size_of_file} GB")

    # TODO: likewise here
    def __len__(self) -> int:
        assert len(self.packet_list) != 0, "No currently loaded partition"
        return self.length

    def __getitem__(self, idx: int) -> Packet:  # CHECK: if I can use floats here
        """
        Retrieves item based on index
        Make sure you have partitioned
        """
        assert (
            len(self.packet_list) != 0
        ), "Capture file is not loaded. Make sure you partition."
        return self.packet_list[idx]


# TODO: You might want to create a parent class for these two in case you use them interchangeably.
class CheapCaptureReader:
    """
    Warning: I can't recommend the use of this class since its  I/O operations will be significantly
    more taxing to total execution time when compared to loading everything at once.
    Regardless if your memory constraints are tight it might be helpful.


    Class will take a large pcap file and partition it into MAX_MEM pcap files
    It will also enable use to give it a time, find the corresopnding file and load it to memory

    Naming
    ------
        lf: large file
    """

    def __init__(self, src_path: Path, dst_dir: Path):
        assert str(src_path).endswith(
            ".pcap"
        ), f"Provided path to capture is not a pcap file"
        self.logger = setup_logger(
            "CaptureReader",
        )
        self.cur_cptfile_ptr = PacketList()  # Empty
        self.cur_cptfile_name = ""

        self.parttn_info = {}  # Filled in when self.parition()
        self.lf_location = src_path
        self.lf_size = src_path.stat().st_size
        self.dst_dir = dst_dir
        self._cache_init()

    def _cache_init(self):
        """
        Logical Convenience Unit for loading cache.
        if dst_dir exists.
            We assume file needed partitioning -> we load it.
        else
            check if passed file needs partition
            if not just it as is
        """
        self.should_partition = self.lf_size > MAX_MEM
        num_files = len(list(self.dst_dir.glob("*.pcap")))
        if num_files > 0:
            # Find the json info object
            json_info = self.dst_dir / "info.json"
            self.should_partition = False
            with open(json_info, "r") as f:
                self.parttn_info = json.load(
                    f
                )  # CHECK: Is it loading it as a dict immediately?
                amnt_of_paritions = len(self.parttn_info["captures"])
                self.logger.debug(f"We have loaded {amnt_of_paritions} partitions")
                self.cur_cptfile_name = "capture_0.pcap"
                self.cur_cptfile_ptr = rdpcap(
                    "./partitions/capture_0.pcap"
                )  # HACK: hardcoded
        else:
            self.dst_dir.mkdir(parents=True, exist_ok=True)
            # Make decision on whether to partition or not
            if not self.should_partition:
                self.cur_cptfile_ptr = rdpcap(str(self.lf_location))
                self.cur_cptfile_name = self.lf_location
            else:  # Run parition when you want
                self.logger.info(
                    "🛑 Pointer to your large capture file is held."
                    "Make sure you run captureReaderInst.partition()"
                )

    @property
    def last_sniff_time(self):
        assert len(self.parttn_info) != 0, "Partitions not loaded"
        return list(self.parttn_info["captures"].values())[-1]["last_sniff_time"]

    # TODO: likewise here
    @property
    def first_sniff_time(self):
        assert len(self.parttn_info) != 0, "Partitions not loaded"
        return list(self.parttn_info["captures"].values())[0]["last_sniff_time"]

    def __len__(self) -> int:
        assert len(self.cur_cptfile_ptr) != 0, "No currently loaded partition"
        return list(self.parttn_info["captures"].values())[-1]["last_idx"] + 1

    @property
    def last_sniff_time(self) -> float:  # CHECK: DTF
        assert len(self.parttn_info) != 0
        return list(self.parttn_info["captures"].values())[-1]["last_sniff_time"]

    def partition(self) -> None:
        """
        Paritions the big file. Better if use calls it themselves so as to not inadvertedly
        load the capture file automatically.
        """
        if len(self.cur_cptfile_ptr) != 0:
            self.logger.warn("You have partitioned already. Will skip")
            return None

        if not self.should_partition:
            self.logger.warn(f"Will not partition, file within {MAX_MEM} budget")
            return

        pcap_rdr = PcapReader(str(self.lf_location))  # type : igore
        amnt_of_paritions = ceil(self.lf_size / MAX_MEM)

        # IMPORTANT: Dictionary below will contain info you need to know how to
        # parse through partitions.
        self.parttn_info = {
            "src_file_loc": str(self.lf_location),
            "src_file_size": self.lf_size,
            "amnt_of_paritions": amnt_of_paritions,
            "partition_size_max": MAX_MEM,
            "captures": {},
        }

        # Read each packet inside self.capture_ptr until we reach 1g
        inter_ptt_bar = tqdm(
            total=amnt_of_paritions, desc="Partitioning Files", leave=True, position=0
        )
        ptt_first_idx = 0
        ptt_last_idx = 0
        for i in range(amnt_of_paritions):
            size_of_file = 0
            intra_ptt_bar = tqdm(
                total=MAX_MEM, desc=f"Partition {i}", leave=False, position=1
            )
            packet = pcap_rdr.read_packet()
            # CHECK: Json is not letting me dump `Decimal` types which
            #   packet.time is returning. Im afraid this might get in the way of precision.
            #   To get this draft finished Ill let it go for now but be careful
            #   Ill mark the two lines of relevant with "DTF"(decimal-to-float)
            earliest_sniff = float(packet.time)  # CHECK: DTF

            packets = []
            while packet != None:
                packets.append(packet)
                size_of_file += len(packet)
                ptt_last_idx += 1
                intra_ptt_bar.update(len(packet))
                if size_of_file > MAX_MEM:
                    break
                else:  # Update earliest sniff time
                    try:
                        packet = pcap_rdr.read_packet()
                    except EOFError:  # EOF
                        break
            # write packets to pcap
            wrpcap(str(self.dst_dir / f"capture_{i}.pcap"), packets)

            self.parttn_info["captures"][f"capture_{i}.pcap"] = {
                "size_of_file": size_of_file,
                "first_sniff_time": earliest_sniff,
                "last_sniff_time": float(packet.time),  # CHECK: DTF
                "first_idx": ptt_first_idx,
                "last_idx": ptt_last_idx - 1,
            }

            ptt_first_idx = ptt_last_idx
            inter_ptt_bar.update(1)

        # Save for later use
        with open(str(self.dst_dir / "info.json"), "w") as f:
            json.dump(self.parttn_info, f)

        self.logger.debug(
            f"Done Paritioning Files. You may find them at {self.dst_dir}"
        )

    def __getitem__(self, idx: int) -> Packet:  # CHECK: if I can use floats here
        """
        Retrieves item based on index
        Make sure you have partitioned
        """
        assert (
            len(self.parttn_info) != 0
        ), "Capture file is not loaded. Make sure you partition."
        pattn, name = self._get_pattn(idx)
        local_idx = idx - self.parttn_info["captures"][name]["first_idx"]
        return pattn[local_idx]

    def _get_pattn(self, idx: int) -> Tuple[PacketList, str]:
        """
        Find parition file containing packet index idx
        """
        for ptt_name, vals in self.parttn_info["captures"].items():
            if idx >= vals["first_idx"] and idx < vals["last_idx"]:
                if self.cur_cptfile_name == ptt_name:
                    return (
                        self.cur_cptfile_ptr,
                        self.cur_cptfile_name,
                    )  # FIX: fix these ambivalent return
                else:
                    # Remove currently loaded file and load new one
                    self.cur_cptfile_ptr = rdpcap(str(self.dst_dir / ptt_name))
                    self.cur_cptfile_name = ptt_name
                    return self.cur_cptfile_ptr, self.cur_cptfile_name
        self.logger.error(f"Could not find time in capture file with idx {idx}")
        exit(-1)  # HACK: ugly


class DynamicWindowSampler:
    def __init__(self, path: str):
        assert Path(path).exists()
        self.logger = setup_logger(__class__.__name__, logging.DEBUG)

        # To be filled later
        self.window_skip = 0.0
        self.window_length = 0.0

        self.logger.info(f"Loading the capture file {path}")
        self.caprdr = CaptureReader(Path(path))

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
