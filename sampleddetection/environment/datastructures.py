from typing import List, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from ..datastructures.flowsession import SampledFlowSession


class State(NamedTuple):
    time_point: float  # DEBUG: Probably only for debug. Otherwise memorization might happen
    cur_window_skip: float
    cur_window_length: float
    flow_sesh: SampledFlowSession
    # Not all features in flow_sesh are to be observed
    observable_features = List[str]

    def as_tensor(self, relevant_datapoints: List[str]) -> Tensor:
        data = self.flow_sesh.get_data()
        # Add all datapoints to a tensor:
        # TODO: think about states where now flows have been onserved
        num_flows = len(data.keys())
        assert num_flows > 0, "We are not ready for non-observations"
        num_features = len(relevant_datapoints)
        tensor = torch.zeros(num_flows, num_features)
        for i, (k, val_dict) in enumerate(data.items()):
            tensor[i] = torch.tensor(
                [float(val_dict[dp]) for dp in relevant_datapoints]
            )
        # CHECK: That aggregation is best policy to vectorize state
        final_state = torch.mean(tensor, dim=0)
        return final_state


class Action(torch.Tensor):
    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, list):
            assert len(data[0]) == 2, "Data should have exactly two columns"
        elif isinstance(data, torch.Tensor):
            assert data.shape[1] == 2, "Data should have exactly two columns"
        return super().__new__(cls, data, *args, **kwargs)

    @property
    def winlength_delta(self):
        return self[:, 0]

    @property
    def winskip_delta(self):
        return self[:, 1]
