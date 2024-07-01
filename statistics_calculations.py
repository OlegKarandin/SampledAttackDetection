"""
This file is more so to try to replicate the statistics obtained from CICFLOWMeter 
but with a smaller file contianing only the necessary data.
"""

import logging
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scapy.all import Packet, PcapReader, rdpcap, wrpcap
from scapy.plist import PacketList
from tqdm import tqdm

# TODO: Clean this mess
# from networking.readers.reader
# from sampleddetection.readers import CSVPacketReader
# from sampleddetection.samplers.window_sampler import DynamicWindowSampler
# from sampleddetection.statistics.window_statistics import get_flows

logger = logging.getLogger("MAIN")
logger.setLevel(logging.INFO)
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatterSciNotation

logger.addHandler(logging.StreamHandler())
from argparse import ArgumentParser

import debugpy

# Catch keyboard inderrupt here:


def argies():
    ap = ArgumentParser()
    ap.add_argument(
        "--src_file",
        default="./data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
    )
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
    csv_reader = CSVReader(args.src_file)
    sampler = DynamicWindowSampler(csv_reader, args.window_length)

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


def distribution_of_iat_across_all_packets(csv_path: str):
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    # Show column names
    # Just collect and plot
    iat: np.ndarray = df[" Flow IAT Mean"].values  # type: ignore
    iat = iat[np.isfinite(iat) & (iat >= 0)]
    dur: np.ndarray = df[" Flow Duration"].values  # type: ignore
    dur = dur[np.isfinite(dur) & (dur >= 0)]
    packets_pers: np.ndarray = df[" Flow Packets/s"].values  # type: ignore
    packets_pers = packets_pers[np.isfinite(packets_pers) & (packets_pers >= 0)]

    # Now plot them in  3 equidistant (triangle) subplots. All being histograms
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    minx, maxx = (np.nanmin(iat), np.nanmax(iat))
    minlx, maxlx = (np.log10(minx), np.log10(maxx))
    print(f"Limits for Flow Mean IAT are [{minx},{maxx}]")
    axs[0].hist(
        iat,
        bins=[b for b in np.logspace(minlx - 1e-3, maxlx, 100)],
        color="blue",
        alpha=0.7,
    )
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xticks([i for i in np.logspace(minlx, maxlx, 10)])
    axs[0].set_xticklabels([f"{i:.1e}" for i in np.logspace(minlx, maxlx, 10)])
    axs[0].set_title("Flow IAT Mean")

    minx, maxx = (np.nanmin(dur), np.nanmax(dur))
    print(f"Limits for Flow Duration are [{minx},{maxx}]")
    axs[1].hist(
        dur,
        bins=[b for b in np.logspace(minlx - 1e-3, maxlx, 100)],
        color="green",
        alpha=0.7,
    )
    axs[1].set_title("Flow Duration")
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_xticks([i for i in np.logspace(minlx, maxlx, 10)])
    axs[1].set_xticklabels([f"{i:.1e}" for i in np.logspace(minlx, maxlx, 10)])

    minx, maxx = (np.nanmin(packets_pers), np.nanmax(packets_pers))
    print(f"Limits for Flow Packets/s are [{minx},{maxx}]")
    axs[2].hist(
        packets_pers,
        bins=[b for b in np.logspace(minlx - 1e-3, maxlx, 100)],
        color="red",
        alpha=0.7,
    )
    axs[2].set_title("Flow Packets/s")
    axs[2].set_yscale("log")
    axs[2].set_xscale("log")
    axs[2].set_xticks([i for i in np.logspace(minlx, maxlx, 10)])
    axs[2].set_xticklabels([f"{i:.1e}" for i in np.logspace(minlx, maxlx, 10)])
    plt.show()


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

    distribution_of_iat_across_all_packets(args.src_file)
    # old tests
    # if False:
    #     single_flow_stats_pcap(
    #         "./bigdata/Wednesday-WorkingHours.pcap",
    #         "40.83.143.209",
    #         "192.168.10.14",
    #         443,
    #         49461,
    #     )
    # else:
    #     single_flow_stats(
    #         "./bigdata/Wednesday.csv",
    #         "40.83.143.209",
    #         "192.168.10.14",
    #         443,
    #         49461,
    #     )
