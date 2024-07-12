# TODO:
# - I have not checked on sub-microsecond precision.
# - We are not dealing with ipv6
import ast
import json
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, List

import debugpy
import numpy as np
import pandas as pd
import pytz
from scapy.all import Packet, PcapNgReader, PcapReader, rdpcap, wrpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from tqdm import tqdm

from sampleddetection.datastructures.context.packet_flow_key import (
    get_packet_flow_key,
    get_simple_tuple,
)
from sampleddetection.datastructures.packet_like import PacketLike, ScapyPacket
from sampleddetection.utils import FLAGS_TO_VAL

# DEBUG: for counting
ipv6_counter = 0
# In order to reduce space I will place TCP as one of the flags, if set to False then its UDP


def argsies():
    ap = ArgumentParser()
    ap.add_argument("--pcap_path", default="./bigdata/Wednesday-WorkingHours.pcap")
    ap.add_argument("--csv_path", default="./bigdata/Wednesday.csv")
    ap.add_argument(
        "--filter",
        type=str,
        default="{}",
        help="String that should be able to be literally evaluated as dictionary for filtering",
    )
    ap.add_argument("-d", "--debug", action="store_true")
    ap.add_argument("--labels", default="./data/label_data.json")
    ap.add_argument(
        "--columnns",
        default=[
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "timestamp",
            "packet_length",
            "int_head_len",
            "int_ttl",
            "layers",
            "tcp_window",
            "payload_size",
            "flags",
            "label",
        ],
    )
    ap.add_argument("--bar_update", default=1000)

    return ap.parse_args()


def read_pcap_global_header(pcap_file_path):
    """Probably useless"""
    with open(pcap_file_path, "rb") as f:
        global_header = f.read(24)  # Read the 24-byte global header

    # Unpack the first 4 bytes to get the magic number
    magic_number = global_header[:4]

    # Check the magic number to determine the timestamp precision
    if magic_number in (b"\xa1\xb2\xc3\xd4", b"\xd4\xc3\xb2\xa1"):
        print("Timestamp precision: microseconds")
    elif magic_number in (b"\xa1\xb2\x3c\x4d", b"\x4d\x3c\xb2\xa1"):
        print("Timestamp precision: nanoseconds")
    else:
        print("Unknown PCAP format")


def filter_pass(pcklike: PacketLike, filter: Dict[str, Any]):
    for k, v in filter.items():
        # Use getattr to get the property of pcklike using the key k
        property_value = getattr(pcklike, k, None)

        # Now you can compare property_value with v or do whatever you need
        if property_value != v:
            return False  # or handle the mismatch as needed
    return True  # If all properties match the filter values


def filter_flow(packet: PacketLike, filter: Dict[str, Any]):
    # Assume flow keys are here
    reverse_dict = {
        "src_ip": filter["dst_ip"],
        "dst_ip": filter["src_ip"],
        "src_port": filter["dst_port"],
        "dst_port": filter["src_port"],
    }
    return filter_pass(packet, filter) or filter_pass(packet, reverse_dict)


def get_packet_label(label: dict, packet: PacketLike, day: str) -> str:
    """Label as in classification label"""
    for k, a in label[day].items():
        if (
            packet.time >= a["start_time"]
            and packet.time <= a["end_time"]
            and packet.src_ip == a["attacker"]
            and packet.dst_ip == a["victim_local"]
        ):
            return k

    return "Benign"


def show_timewindows(attacks: Dict, timezone: str = "Etc/GMT+3"):
    print("Your time windows will be:")
    utc_minus_3 = pytz.timezone(timezone)
    for attack, specs in attacks.items():
        start = specs["start_time"]
        end = specs["end_time"]

        # Conver to UTC-3
        start_humanformat = datetime.utcfromtimestamp(int(start))
        end_humanformat = datetime.utcfromtimestamp(int(end))

        utc3_start = (
            start_humanformat.replace(tzinfo=pytz.utc)
            .astimezone(utc_minus_3)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
        utc3_end = (
            end_humanformat.replace(tzinfo=pytz.utc)
            .astimezone(utc_minus_3)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
        print(f"{attack} Window: \n\tFr: {utc3_start}\n\tTo: {utc3_end}")


def packet_parser(
    packet: Packet, filter_dict: Dict[str, Any], labels: Dict
) -> List[Any]:
    """
    Returns row that will be written into the csv file.
    """
    if not (TCP in packet) and not (UDP in packet):
        return []  # CHECK: if this is correct

    global ipv6_counter
    data_protocol = "TCP" if "TCP" in packet else "UDP"

    packet_like = ScapyPacket(packet)

    # flag_bits.setall(False)
    if "IPv6" in packet:
        ipv6_counter = ipv6_counter + 1
        return []  # We are not dealing with ipv6 (See original CICFlowMeter)

    # For now we assume we are only filtering flows
    if len(filter_dict) != 0 and not filter_flow(packet_like, filter_dict):
        return []

    # flag_bits = np.zeros(len(FLAGS_TO_VAL.values()), dtype=bool)
    flag_dict = {k: False for k, _ in FLAGS_TO_VAL.items()}
    src_ip, dst_ip, srcp, dstp = get_simple_tuple(ScapyPacket(packet))

    # TODO: theres likely an easier way to do this
    if "TCP" in packet:
        for i, (k, v) in enumerate(FLAGS_TO_VAL.items()):
            if packet["TCP"].flags & v != 0:
                flag_dict[k] = True

    layers = [str(l.__name__) for l in packet.layers()]
    # Append data
    row = [
        src_ip,
        dst_ip,
        srcp,
        dstp,
        data_protocol,
        packet.time,
        len(packet),
        packet["IP"].ihl,
        packet["IP"].ttl,
        layers,
        packet["TCP"].window if data_protocol == "TCP" else 0,
        len(packet[data_protocol].payload),
        str(flag_dict),
        get_packet_label(labels, packet_like, "Wednesday"),
    ]
    return row


def get_tresol():
    # Load the pcap file
    caprdr = PcapNgReader(str(args.pcap_path))  # type: ignore
    packet = caprdr.read_packet()

    # Access the Interface Description Blocks (IDBs)
    # Iterate over the IDBs and print the timestamp resolution for each interface

    assert (
        len(caprdr.interfaces) == 1
    ), "Script has not been writte to handle over 1 interface"
    interface = caprdr.interfaces[0]

    # Extract the if_tsresol option (if it exists)
    print(
        f"PCAP's interface information:"
        f"\n  Link Type {interface[0]}"
        f"\n  Snap len {interface[1]}"
        f"\n  TSResol {interface[2]}"
    )
    # Calculate actual resolution
    denominator = interface[2]
    return denominator


if __name__ == "__main__":
    # Start Here
    args = argsies()
    if args.debug:
        print("Waitinf for clientin port 42019")
        debugpy.listen(42019)
        debugpy.wait_for_client()
        print("Client connecting. Debugging...")

    # Get Labels dictionary(json_file)
    labels = json.load(open(args.labels, "r"))

    # Get first four bytes of args.pcap_path
    magic_bytes = open(args.pcap_path, "rb").read(4)
    parsing_metadata = {"magin_bytes": "0x" + magic_bytes.hex()}
    # Check for the precision of the timestamp
    resolution = get_tresol()
    parsing_metadata["time_resol(denominator)"] = resolution

    # Ensure Label Data is correct
    show_timewindows(labels["Wednesday"])

    caprdr = PcapNgReader(str(args.pcap_path))  # type: ignore
    cur_pack = caprdr.read_packet()
    all_rows = []
    bar = tqdm(total=13788878, desc="Reading packets")
    filter_dict = ast.literal_eval(args.filter)

    # Change this if necessary on your end
    i = 0
    utc_minus_3 = pytz.timezone("Etc/GMT+3")
    try:
        while cur_pack != None:
            # TODO create an index for flows as well.

            # 表示過不過
            row = packet_parser(cur_pack, filter_dict, labels)
            if len(row) != 0:
                all_rows.append(row)

            cur_time = int(cur_pack.time)
            cur_time_human = datetime.utcfromtimestamp(int(cur_time))
            local_datetime = (
                cur_time_human.replace(tzinfo=pytz.utc)
                .astimezone(utc_minus_3)
                .strftime("%Y-%m-%d %H:%M:%S")
            )

            if (i % args.bar_update) == 0:
                bar.set_description(
                    f"Reading packets (Time {local_datetime})(Added {i} rows)"
                )

            bar.update(1)
            i += 1
            try:
                cur_pack = caprdr.read_packet()
            except EOFError:
                break
    except KeyboardInterrupt:
        print(
            "Was interrupted by keyboard will write what we managed to add to the list."
        )
    # Now we turn it into a DataFrame and save it

    print("All records received, now turning into csv_path")
    print(f"Skipped {ipv6_counter} ipv6 packets")

    new_df = pd.DataFrame(all_rows, columns=args.columnns)
    new_df = new_df.sort_values(by="timestamp")
    new_df.to_csv(args.csv_path, index=False)

    with open(args.csv_path.replace(".csv", ".json"), "w") as f:
        json.dump(parsing_metadata, f, indent=4)
