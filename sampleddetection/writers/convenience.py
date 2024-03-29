import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

from sampleddetection.datastructures.context.packet_flow_key import FLOW_KEY_NAMES
from sampleddetection.datastructures.flow import Flow
from sampleddetection.datastructures.flowsession import SampledFlowSession

from ..common_lingo import STRING_TO_ATTACKS, Attack


def save_to_csv(
    sessions: List[SampledFlowSession],
    path: str,
    desired_features: List[str],
    samples_per_class: Dict[Attack, int],
    overwrite: bool = False,
):
    # Ensure path does not exist
    assert (
        overwrite or Path(path).exists() is False
    ), f"Given path {path} already exists, delete before running"
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)

    class_count = {k: 0 for k, _ in samples_per_class.items()}

    columns = FLOW_KEY_NAMES + ["count"] + desired_features
    csv_file = open(path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(columns)
    num_rows = 0

    for session in sessions:
        multiflow_data = session.get_data()
        for flowk_wcont, flow_stats in multiflow_data.items():
            flowk, count = flowk_wcont
            ordered_vals = [flow_stats[f] for f in desired_features]
            flow_enum = (
                Attack.GENERAL
                if STRING_TO_ATTACKS[flow_stats["label"]] != Attack.BENIGN
                else Attack.BENIGN
            )
            class_count[flow_enum] += 1

            if class_count[flow_enum] < samples_per_class[flow_enum]:
                csv_writer.writerow(flowk + (count,) + tuple(ordered_vals))
            num_rows += 1

    csv_file.close()


def save_flows_to_csv(
    flows: List[Tuple[Tuple[Tuple, int], Flow]],  # I know I plan to fix this
    path: str,
    desired_features: List[str],
    samples_per_class: Dict[Attack, int],
    overwrite: bool = False,
):
    # Write the same exact function as above but now using flows
    assert (
        overwrite or Path(path).exists() is False
    ), f"Given path {path} already exists, delete before running"
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)

    class_count = {k: 0 for k, _ in samples_per_class.items()}

    columns = FLOW_KEY_NAMES + ["count"] + desired_features
    csv_file = open(path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(columns)
    num_rows = 0

    for flow_key, flow in flows:
        flow_data = flow.get_data()
        flow_def, count = flow_key
        ordered_vals = [flow_data[f] for f in desired_features]
        flow_enum = Attack.GENERAL if flow.label != Attack.BENIGN else Attack.BENIGN
        class_count[flow_enum] += 1

        if class_count[flow_enum] < samples_per_class[flow_enum]:
            csv_writer.writerow(flow_def + (count,) + tuple(ordered_vals))
        num_rows += 1

    csv_file.close()
