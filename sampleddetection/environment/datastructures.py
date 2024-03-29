from typing import List, NamedTuple

from ..datastructures.flowsession import SampledFlowSession


class State(NamedTuple):
    time_point: float
    window_skip: float
    window_length: float
    flow_sesh: SampledFlowSession

    def sesh_to_tensor(self, relevant_datapoints: List[str]):
        data = self.flow_sesh.get_data()


class Action(NamedTuple):
    winlength_delta: float
    winskip_delta: float
