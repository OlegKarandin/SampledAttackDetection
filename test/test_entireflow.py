"""
All the hardcoded statistics used in this script where 
calculated by hand.
"""
import ast
import os

# Add parent dir to include dir
import sys
from pathlib import Path

import pandas as pd
import pytest

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(f"Parent path is {parent_path}")
sys.path.append(parent_path)

from logging import DEBUG as LVL_DEBUG
from logging import INFO as LVL_INFO

from sampleddetection.datastructures.flowsession import SampledFlowSession
from sampleddetection.utils import NpEncoder, setup_logger

global logger
logger = setup_logger(os.path.basename(__file__), LVL_DEBUG)

import subprocess

from sampleddetection.readers.readers import CSVReader
from sampleddetection.samplers.window_sampler import (
    DynamicWindowSampler,
    UniformWindowSampler,
)


def test_flowbytes(flowsesh: SampledFlowSession):
    data = flowsesh.get_data()
    first_key = next(iter(data))
    bytes_length = (
        data[first_key]["totlen_fwd_pkts"] + data[first_key]["totlen_bwd_pkts"]
    )

    assert bytes_length == 142114  # DATA extracted from wireshark


def test_flowpackets(flowsesh: SampledFlowSession):
    data = flowsesh.get_data()
    first_key = next(iter(data))
    packet_count = data[first_key]["tot_fwd_pkts"] + data[first_key]["tot_bwd_pkts"]
    assert packet_count == 172  # DATA extracted from wireshark


def test_flow_duration(flowsesh: SampledFlowSession):
    """Units are in Microseconds"""
    data = flowsesh.get_data()
    first_key = next(iter(data))
    flowduration = data[first_key]["flow_duration"]
    assert (
        pytest.approx(flowduration, rel=1e-3) == 91501624.1074
    )  # DATA extracted from wireshark


def test_bytes_rate(flowsesh: SampledFlowSession):
    """Rate is Bytes/Second"""
    data = flowsesh.get_data()
    first_key = next(iter(data))
    bytes_rate = data[first_key]["flow_byts_s"]
    logger.info(
        f"Bytes rate is {bytes_rate} bytes/sec and truth is {142114 / 91.5016241074}"
    )
    assert pytest.approx(bytes_rate, abs=1e-6) == (142114 / 91.5016241074)


def test_pkts_rate(flowsesh: SampledFlowSession):
    data = flowsesh.get_data()
    first_key = next(iter(data))
    pkts_rate = data[first_key]["flow_pkts_s"]
    logger.info(
        f"Packets rate is {pkts_rate} pkts/sec and truth isi {(172 / 91.5016241074)}"
    )
    assert pytest.approx(pkts_rate, abs=1e-6) == (172 / 91.5016241074)
    """Rate is Bytes/Second"""


def test_header_length(flowsesh: SampledFlowSession, pcap_path: str):
    # Run
    cmd = (
        "tshark -r "
        + pcap_path
        + "  -Y 'ip.hdr_len' -T fields -e ip.hdr_len | awk '{sum += $1} END {print sum}' "
    )
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )

    # Check if the commands were successful
    if result.returncode != 0:
        raise Exception(f"Command failed with return code {result.returncode}")

    truth = int(result.stdout.strip())

    data = flowsesh.get_data()
    first_key = next(iter(data))
    hdr_len = data[first_key]["fwd_header_len"] + data[first_key]["bwd_header_len"]

    # For some reason scapy stores it this way
    assert hdr_len * 4 == truth


def test_mean_interarrivaltime(flowsesh: SampledFlowSession, pcap_path: str):
    cmd = " tshark -r ./bigdata/specific_instances/40.83.143.209_192.168.10.14_443_49461.pcap  -T fields -e frame.time_epoch | awk 'NR > 1 {print $1 - prev} {prev = $1}' | awk '{sum += $1; count++} END {print sum / count}'"
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    if result.returncode != 0:
        logger.error(f"Subcommand returned with error: {result}")

    # Turn to microseconds
    truth = float(result.stdout.strip()) * 1e6  # We are dealing with microseconds

    data = flowsesh.get_data()
    first_key = next(iter(data))
    mean_iat = data[first_key]["flow_iat_mean"]
    logger.debug(f"Truth is {truth} and result is {mean_iat}")
    assert pytest.approx(mean_iat, abs=1e-0) == truth


def test_forward_iat(flowsesh: SampledFlowSession, pcap_path: str):
    cmd = " tshark -r ./bigdata/specific_instances/40.83.143.209_192.168.10.14_443_49461.pcap  -Y \" ip.dst == 40.83.143.209 \" -T fields -e frame.time_epoch | awk 'NR > 1 {print $1 - prev} {prev = $1}' | awk '{sum += $1; count++} END {print sum / count}'"
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    if result.returncode != 0:
        logger.error(f"Subcommand return with error {result}")
    data = flowsesh.get_data()
    truth = float(result.stdout.strip()) * 1e6
    first_key = next(iter(data))
    forward_mean_iat = data[first_key]["fwd_iat_max"]

    assert pytest.approx(forward_mean_iat, 1e0) == truth


def test_backward_iat(flowsesh: SampledFlowSession, pcap_path: str):
    cmd = " tshark -r ./bigdata/specific_instances/40.83.143.209_192.168.10.14_443_49461.pcap  -Y \" ip.dst == 192.168.10.14 \" -T fields -e frame.time_epoch | awk 'NR > 1 {print $1 - prev} {prev = $1}' | awk '{sum += $1; count++} END {print sum / count}'"
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    if result.returncode != 0:
        logger.error(f"Subcommand return with error {result}")
    data = flowsesh.get_data()
    truth = float(result.stdout.strip()) * 1e6
    first_key = next(iter(data))
    forward_mean_iat = data[first_key]["bwd_iat_max"]

    assert pytest.approx(forward_mean_iat, 1e0) == truth


def test_flag_count(flowsesh: SampledFlowSession, pcap_path: str):
    cmd = """
    tshark -r ./bigdata/specific_instances/40.83.143.209_192.168.10.14_443_49461.pcap -Y "tcp.flags" -T \
    fields -e tcp.flags.syn -e tcp.flags.fin -e tcp.flags.reset -e tcp.flags.push -e tcp.flags.ack -e tcp.flags.urg -e tcp.flags.ecn -e tcp.flags.cwr \
    | awk '
        {
            syn_count += $1;
            fin_count += $2;
            rst_count += $3;
            psh_count += $4;
            ack_count += $5;
            urg_count += $6;
            ecn_count += $7;
            cwr_count += $8;
        }
        END {
            printf("{");
            printf("\\"FIN\\": %d, ", fin_count);
            printf("\\"SYN\\": %d, ", syn_count);
            printf("\\"RST\\": %d, ", rst_count);
            printf("\\"PSH\\": %d, ", psh_count);
            printf("\\"ACK\\": %d, ", ack_count);
            printf("\\"URG\\": %d, ", urg_count);
            printf("\\"ECE\\": %d, ", ecn_count);
            printf("\\"CWR\\": %d", cwr_count);
            printf("}\\n");
        }'
        """
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    if result.returncode != 0:
        logger.error(f"Subcommand return with error {result}")
        assert False, "Subcommand exited with error"

    logger.debug(f"Truth is {result.stdout.strip()}")
    truths = ast.literal_eval(result.stdout.strip())

    data = flowsesh.get_data()
    first_key = next(iter(data.keys()))

    estimations = {
        "FIN": data[first_key]["fin_flag_cnt"],
        "SYN": data[first_key]["syn_flag_cnt"],
        "RST": data[first_key]["rst_flag_cnt"],
        "PSH": data[first_key]["psh_flag_cnt"],
        "ACK": data[first_key]["ack_flag_cnt"],
        "URG": data[first_key]["urg_flag_cnt"],
        "ECE": data[first_key]["ece_flag_cnt"],
        "CWR": data[first_key]["cwr_flag_cnt"],
    }
    logger.debug(f"Estimations is {str(estimations)}")
    for k, v in estimations.items():
        assert v == truths[k], f"{k} flag count is incorrect"


def test_has_flag(flowsesh: SampledFlowSession, pcap_path: str):
    cmd = """
    tshark -r ./bigdata/specific_instances/40.83.143.209_192.168.10.14_443_49461.pcap -Y "tcp.flags" -T \
    fields -e tcp.flags.syn -e tcp.flags.fin -e tcp.flags.reset -e tcp.flags.push -e tcp.flags.ack -e tcp.flags.urg -e tcp.flags.ecn -e tcp.flags.cwr \
    | awk '
        {
            syn_count += $1;
            fin_count += $2;
            rst_count += $3;
            psh_count += $4;
            ack_count += $5;
            urg_count += $6;
            ecn_count += $7;
            cwr_count += $8;
        }
        END {
            printf("{");
            printf("\\"FIN\\": %d, ", fin_count);
            printf("\\"SYN\\": %d, ", syn_count);
            printf("\\"RST\\": %d, ", rst_count);
            printf("\\"PSH\\": %d, ", psh_count);
            printf("\\"ACK\\": %d, ", ack_count);
            printf("\\"URG\\": %d, ", urg_count);
            printf("\\"ECE\\": %d, ", ecn_count);
            printf("\\"CWR\\": %d", cwr_count);
            printf("}\\n");
        }'
        """
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    if result.returncode != 0:
        logger.error(f"Subcommand return with error {result}")
        assert False, "Subcommand exited with error"

    logger.debug(f"Truth is {result.stdout.strip()}")
    truths = ast.literal_eval(result.stdout.strip())
    truths = {k: v != 0 for k, v in truths.items()}

    data = flowsesh.get_data()
    first_key = next(iter(data.keys()))

    estimations = {
        "FIN": data[first_key]["fin_has_flag"],
        "SYN": data[first_key]["syn_has_flag"],
        "RST": data[first_key]["rst_has_flag"],
        "PSH": data[first_key]["psh_has_flag"],
        "ACK": data[first_key]["ack_has_flag"],
        "URG": data[first_key]["urg_has_flag"],
        "ECE": data[first_key]["ece_has_flag"],
        "CWR": data[first_key]["cwr_has_flag"],
    }
    logger.debug(f"Estimations is {str(estimations)}")
    for k, v in estimations.items():
        assert v == truths[k], f"{k} flag count is incorrect"


def test_downup_ratio(flowsesh: SampledFlowSession, pcap_path: str):
    file_name_only = os.path.basename(pcap_path)
    logger.debug(f"Ds wierd {file_name_only.replace('.pcap', '').split('_')}")
    src_ip, dst_ip, src_port, dst_port = file_name_only.replace(".pcap", "").split("_")

    cmdforward = f"tshark -r {pcap_path} -Y 'tcp and ip.src=={src_ip} and ip.dst=={dst_ip} and tcp.srcport=={src_port} and tcp.dstport=={dst_port}' | wc -l"
    cmdbackward = f"tshark -r {pcap_path} -Y 'tcp and ip.src=={dst_ip} and ip.dst=={src_ip} and tcp.srcport=={dst_port} and tcp.dstport=={src_port}' | wc -l"

    resultfwd = subprocess.run(
        cmdforward,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    resultbwd = subprocess.run(
        cmdbackward,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )

    if resultfwd.returncode != 0:
        logger.error(f"Subcommand returned with error: {resultfwd}")

    if resultbwd.returncode != 0:
        logger.error(f"Subcommand returned with error: {resultbwd}")

    truth = int(resultbwd.stdout.strip()) / int(resultfwd.stdout.strip())

    flowsesh.get_data()
    first_eg = next(iter(flowsesh.get_data().values()))
    est_ratio = first_eg["down_up_ratio"]

    assert pytest.approx(est_ratio, abs=1e-6) == truth


def test_idle_times(
    flowsesh: SampledFlowSession,
    pcap_path: str,
    active_timeout: float,
    clump_timeout: float,
):
    flow = next(iter(flowsesh.flows.values()))

    filename = os.path.basename(pcap_path)
    test_path = os.path.dirname(os.path.abspath(__file__))

    src_ip, dst_ip, src_port, dst_port = filename.replace(".pcap", "").split("_")

    ident_tuple = ",".join([src_ip, dst_ip, src_port, dst_port])
    inactive_sh_path = os.path.join(test_path, "inactive.sh")
    cmd = f"""
        sh {inactive_sh_path} {pcap_path} {ident_tuple} {active_timeout} {clump_timeout}
        """
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    if result.returncode != 0:
        logger.error(f"Subcommand returned with error: {result}")
    first_eg = next(iter(flowsesh.get_data().values()))
    est_idle_mean = first_eg["idle_mean"]
    # result will give numbers separated by new line, here we take a mean of all of them
    truth = sum([float(x) * 1e6 for x in result.stdout.strip().split("\n")]) / len(
        result.stdout.strip().split("\n")
    )

    assert pytest.approx(est_idle_mean, abs=1e0) == truth


# TODO: I'm just not very sure how helpful this static is or what is their correct use for it.
# def test_window_size(flowsesh: SampledFlowSession, csv_path: str):
#     pass


@pytest.fixture
def flowsesh(request) -> SampledFlowSession:
    csvreader = CSVReader(Path(request.config.getoption("--csvfile")))
    dws = DynamicWindowSampler(csvrdr=csvreader)
    first_sniff = dws.csvrdr.first_sniff_time - 1
    final_sniff = dws.csvrdr.last_sniff_time
    non_existant_window = 1e-16
    flowsesh = dws.sample(
        first_sniff,
        non_existant_window,
        final_sniff - first_sniff + non_existant_window + 1,
    )
    return flowsesh


@pytest.fixture
def csv_path(request) -> str:
    csvreader = request.config.getoption("--csvfile")
    return csvreader


@pytest.fixture
def pcap_path(request) -> str:
    pcappath = request.config.getoption("--pcappath")
    return pcappath


@pytest.fixture
def active_timeout(request) -> int:
    val = request.config.getoption("--active_timeout")
    return val


@pytest.fixture
def clump_timeout(request) -> int:
    val = request.config.getoption("--clump_timeout")
    return val


# @pytest.fixture
# def csvreader(request) -> CSVReader:
#     return CSVReader(Path(request.config.getoption("--csvfile")))


# @pytest.fixture
# def sample(request) -> SampledFlowSession:
#     src_ip =
#     return CSVReader(
