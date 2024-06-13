import json
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from time import time
from typing import List, Tuple, Union

import pandas as pd
from scapy.all import Packet, PcapReader, rdpcap, wrpcap
from scapy.plist import PacketList
from tqdm import tqdm

from networking.datastructures.packet_like import CSVPacket
from sampleddetection.datastructures import CSVSample
from sampleddetection.readers import AbstractTimeSeriesReader
from sampleddetection.utils import setup_logger

MAX_MEM = 15e9  # GB This is as mcuh as we want in ram at once


class NetCSVReader(AbstractTimeSeriesReader):
    """
    Interface for CSV but with Packets as Samles
    Expectations:
        - csv must have timestamp column
    """

    def __init__(self, csv_path: Path):
        """
        Parameters
        ~~~~~~~~~~
            mdata_path : Path to store meta data after having a pass over pcap file and forming an index
        """
        # Parameters
        self.logger = setup_logger(__class__.__name__)
        self.csv_path = csv_path
        self.logger.info("Reading csv...")
        strt = time()
        self.csv_df: pd.DataFrame = pd.read_csv(csv_path)
        self.logger.info(
            f"CSV loaded, took {time() - strt: 4.2f} seconds with {len(self.csv_df)} length"
        )

        # TODO: Check this to work properly
        self.first_sniff_time: float = self.csv_df.loc[0, "timestamp"].astype(float)
        self.last_sniff_time: float = self.csv_df.loc[
            self.csv_df.index[-1], "timestamp"
        ].astype(float)

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx_or_slice) -> Union[CSVPacket, List[CSVPacket]]:
        # return self.csv_df.iloc[idx]
        # check if it is a slice
        self.logger.debug(
            f"We are getting argument : {idx_or_slice} of type {type(idx_or_slice)}"
        )
        if isinstance(idx_or_slice, int):
            self.logger.debug(f"We are feeding {self.csv_df.iloc[idx_or_slice]}")
            return CSVPacket(self.csv_df.iloc[idx_or_slice])
        elif isinstance(idx_or_slice, slice):
            # This is operating across th
            return [
                CSVPacket(row) for _, row in self.csv_df.iloc[idx_or_slice].iterrows()
            ]
        else:
            raise TypeError(
                f"Provided CSVReader with neither int nor slice. Unrecognizable type {type(idx_or_slice)}"
            )

    def getTimestamp(self, idx):
        return self.csv_df.iloc[idx]["timestamp"]

    @property
    def init_time(self) -> float:
        return self.csv_df.iloc[0]["timestamp"]

    @property
    def fin_time(self) -> float:
        """
        Final Timestamp
        """
        return self.csv_df.iloc[-1]["timestamp"]


class CaptureReader(AbstractTimeSeriesReader):
    """
    Interface for PCAP File
    """

    def __init__(self, src_path: Path):
        assert str(src_path).endswith(
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

        self._load_data()

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
        start_time = time()
        # packet_list = pcap_rdr.read_all()
        # TODO: Place back the packet thing.
        elapsed_time = time() - start_time
        print(f"Loading this file took {elapsed_time}")

        self.first_sniff_time = float(packet.time)  # CHECK: DTF
        self.logger.info("Attempting to load monolithic file")
        last_packet = packet

        t_bar = tqdm(desc="Loading Monolithic File (GB)", leave=True, position=0)
        pckt_cnt = 0
        while packet != None:
            self.packet_list.append(packet)
            size_of_file += len(packet)
            pckt_cnt += 1
            if pckt_cnt % 1000 == 0:
                t_bar.update(size_of_file / 10e9)
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


class PartitionCaptureReader:
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
        self.cur_cptfile_ptr = rdpcap(str(self.lf_location))  # Empty
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
