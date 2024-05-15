from typing import List, NamedTuple, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
from common_lingo import ArrayLike
from torch import Tensor

from networking.datastructures.flowsession import SampledFlowSession
from sampleddetection.datastructures import Action, State

# Nothing much else to add
NetAction = Action


class NetState(State):

    def __init__(
        self,
        time_point: float,
        window_skip: float,
        cur_window_length: float,
        flow_sesh: SampledFlowSession,
        observable_features: List[str],
    ):
        super().__init__(
            time_point=time_point,
            window_skip=window_skip,
            cur_window_length=cur_window_length,
            observations=flow_sesh.get_data(),
        )
        self.flow_sesh = flow_sesh
        self.observable_features = observable_features

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

    # define method for return this classes properties as a NamedTuple
    def as_tuple(self) -> Tuple:
        return (
            self.time_point,
            self.cur_window_skip,
            self.cur_window_length,
            self.flow_sesh,
            self.observable_features,
        )
