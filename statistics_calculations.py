"""
This file is more so to try to replicate the statistics obtained from CICFLOWMeter 
but with a smaller file contianing only the necessary data.
"""
import logging
from time import time

import pandas as pd
from scapy.all import Packet, PcapReader, rdpcap, wrpcap
from scapy.plist import PacketList
from tqdm import tqdm

from sampleddetection.datastructures.context.packet_flow_key import (
    get_packet_flow_key,
    get_simple_tuple,
)
from sampleddetection.samplers.window_sampler import DynamicWindowSampler
from sampleddetection.statistics.window_statistics import get_flows

logger = logging.getLogger("MAIN")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
from argparse import ArgumentParser

import debugpy

# Catch keyboard inderrupt here:


def argies():
    ap = ArgumentParser()
    ap.add_argument("--src_file", default="./bigdata/Wednesday.csv")
    ap.add_argument(
        "--window_skip", default=2.0, help="The amount of time between observations."
    )
    ap.add_argument(
        "--window_length",
        default=1.0,
        help="The amount of time to observe for pacekts.",
    )
    ap.add_argument(
        "--debug", "-d", action="store_true", help="Launch debugpy's debugging mode"
    )

    return ap.parse_args()


def sampler_test():
    logger.info(f"Quick Head look is :{file.head()}")

    logger.info("Starting with the sampling")
    sampler = DynamicWindowSampler(args.src_file, args.window_skip, args.window_length)

    initial_time = sampler.caprdr.first_time + 1.0
    packet_list: PacketList = sampler.sample(0, initial_time, 3.0)

    logger.info(f"Obtained {len(packet_list)} packets")

    # Then actually calculate the statistics
    logger.info("Packets acquired, calculating corresponding flows")
    flows = get_flows(packet_list)
    logger.info(f"Flows obtained. Totaling {len(flows)}")
    logger.info(f"They look like {flows}")


def single_flow_stats(
    csv_file: str, src_ip: str, dst_ip: str, src_port: int, dst_port: int
):
    """Will retrieve all packets that can map to that tuple"""
    logger.info(f"Opening file {args.src_file}")
    start_time = time()
    csv_reader = pd.read_csv(args.src_file)
    end_time = time() - start_time
    logger.info(
        f"File loaded in {end_time}s. Amount of packets to go through is {len(csv_reader)}"
    )

    filtered_series_left = (
        (csv_reader["src_ip"] == src_ip)
        & (csv_reader["dst_ip"] == dst_ip)
        & (csv_reader["src_port"] == src_port)
        & (csv_reader["dst_port"] == dst_port)
    )
    filtered_series_right = (
        (csv_reader["src_ip"] == dst_ip)
        & (csv_reader["dst_ip"] == src_ip)
        & (csv_reader["src_port"] == dst_port)
        & (csv_reader["dst_port"] == src_port)
    )
    filtered_rows = csv_reader[filtered_series_left | filtered_series_right]
    logger.info(f"Obtained {len(filtered_rows)} filtered_rows ")
    summation = calc_total_bytes(filtered_rows)
    logger.info(f"Summation of row is {summation}")

    tuple_str = f"{src_ip}_{dst_ip}_{src_port}_{dst_port}"
    filtered_rows.to_csv(f"./bigdata/specific_instances/{tuple_str}.csv")


def single_flow_stats_pcap(pcap_file, src_ip, dst_ip, src_port, dst_port):
    """
    Same as above but reading through a whole pcap file packet by packet
    """
    caprdr = PcapReader(str(pcap_file))  # type: ignore

    cur_pack = caprdr.read_packet()
    filtered_packets = []
    bar = tqdm(total=13788878, desc="Filtering packets")

    try:
        while cur_pack != None:
            if "IP" in cur_pack:
                if (
                    cur_pack["IP"].src == src_ip
                    and cur_pack["IP"].dst == dst_ip
                    and cur_pack["TCP"].sport == src_port  # TODO: make it udp inclusive
                    and cur_pack["TCP"].dport == dst_port
                ) or (
                    cur_pack["IP"].src == dst_ip
                    and cur_pack["IP"].dst == src_ip
                    and cur_pack["TCP"].sport == dst_port
                    and cur_pack["TCP"].dport == src_port
                ):
                    filtered_packets.append(cur_pack)
                    bar.set_description(f"Filtered packets({len(filtered_packets)})")
            bar.update(1)
            cur_pack = caprdr.read_packet()
    except KeyboardInterrupt:
        logger.info(
            f"Keyboard interrupt detected,, will containue to save the {len(filtered_packets)} packets found"
        )
    except EOFError:
        logger.info("Reached end of pcap file")
    finally:
        logger.error("Unknown exception caught")

    logger.info(f"Obtained {len(filtered_packets)} filtered_packets ")
    tuple_str = f"{src_ip}_{dst_ip}_{src_port}_{dst_port}"
    wrpcap("./bigdata/specific_instances/" + tuple_str + ".pcap", filtered_packets)


def calc_total_bytes(csv_reader: pd.DataFrame):
    total_bytes = csv_reader["packet_length"]
    summation = total_bytes.sum()
    return summation


if __name__ == "__main__":
    # We load the file
    args = argies()
    # Check for Debug
    if args.debug:
        logger.info("Running in debug mode. Waiting for connection in 42019")
        debugpy.listen(42019)
        debugpy.wait_for_client()
        logger.info("Client connected. Proceeding with debug session.")

    if False:
        single_flow_stats_pcap(
            "./bigdata/Wednesday-WorkingHours.pcap",
            "40.83.143.209",
            "192.168.10.14",
            443,
            49461,
        )
    else:
        single_flow_stats(
            "./bigdata/Wednesday.csv",
            "40.83.143.209",
            "192.168.10.14",
            443,
            49461,
        )
