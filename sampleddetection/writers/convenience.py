import csv
import os
from pathlib import Path
from typing import List

from sampleddetection.datastructures.context.packet_flow_key import FLOW_KEY_NAMES
from sampleddetection.datastructures.flowsession import SampledFlowSession


def save_to_csv(
    sessions: List[SampledFlowSession],
    path: str,
    desired_features: List[str],
    overwrite: bool = False,
):
    # Ensure path does not exist
    assert (
        overwrite or Path(path).exists() is False
    ), f"Given path {path} already exists, delete before running"
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)

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
            csv_writer.writerow(flowk + (count,) + tuple(ordered_vals))
            num_rows += 1

    csv_file.close()
