import json
import logging
from math import ceil
from pathlib import Path
from random import shuffle
from typing import Tuple

import numpy as np
from scapy.all import Packet, PcapReader, rdpcap, wrpcap
from scapy.plist import PacketList
from tqdm import tqdm

from ..utils import setup_logger

# import PcaketList
# MAX_MEM = 4e9  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 250e6  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 100e6  # GB This is as mcuh as we want in ram at once
# MAX_MEM = 2 ** (26.575)
MAX_MEM = 2 ** (23.25)


class CaptureReader:
    """
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
                self.logger.debug(f"We have loaded {amnt_of_paritions}")
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

    # TODO: Fix: this will when there is no partition

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
        assert len(self.cur_cptfile_ptr) == 0, "No currently loaded partition"
        return list(self.parttn_info["captures"].values())[-1]["last_idx"] + 1

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

        pcap_rdr = PcapReader(str(self.lf_location))
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

    def __getitem__(self, idx) -> Packet:  # CHECK: if I can use floats here
        """
        Retrieves item based on index
        Make sure you have partitioned
        """
        assert (
            len(self.parttn_info) != 0
        ), "Capture file is not loaded. Make sure you partition."
        pattn, name = self._get_pattn(idx)
        local_idx = idx - self.parttn_info[name]["name"]["first_idx"]
        return pattn[local_idx]

    def _get_pattn(self, idx: int) -> Tuple[PacketList, str]:
        """
        Find parition file containing packet index idx
        """
        for ptt_name, vals in self.parttn_info["captures"].items():
            if idx > vals["first_idx"] and idx < vals["last_idx"]:
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
        self.logger.error("Could not find time in capture file")
        exit(-1)  # HACK: ugly


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
        self.caprdr = CaptureReader(
            Path(path), Path(partition_loc)  # type:ignore CHECK:
        )
        self.caprdr.partition()

        # Locate and Read the file
        self.logger.info("⚠️ Loading the capture file. This will likely take a while")
        self.first_ts = self.caprdr.first_sniff_time
        self.last_ts = self.caprdr.last_sniff_time
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
            cur_idx = self._binary_search(self.caprdr, rand_time)
            end_time = rand_time + self.window_length

            # cur_packet = self.capture[cur_idx]
            # For each of these initial samples get its window.
            window_packet_list = []
            while self.caprdr[cur_idx].time < end_time:
                # Append Packet
                assert hasattr(  # TODO: remove this if it never really gets called.
                    self.caprdr[cur_idx], "time"
                ), "Apparently packet does not have sniff-timestamp"

                if (
                    "TCP" not in self.caprdr[cur_idx]
                    and "UDP" not in self.caprdr[cur_idx]
                ):
                    continue
                window_packet_list.append(self.caprdr[cur_idx])

                cur_idx += 1
            # When Done capturing we add the packet list:
            windows_list.append(window_packet_list)

        # TODO: Ensure good balance between labels.
        return windows_list

    def _binary_search(self, cap_rdr: CaptureReader, target_time: float):
        """
        Given a capture and a target time, will return the index of the first packet
        that is after the target time
        """
        # Initialize variables
        low = 0
        high = len(cap_rdr)
        mid = 0

        # Mid should be above our target_time.

        # Do binary search
        while high > low:
            mid = ceil(
                (high + low) / 2
            )  # CHECK: It *should* be ceil. Check nonetheless
            if target_time > cap_rdr[mid].time:
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
