from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np
from torch import Tensor


# Just define by properties one expects it to have
class StateLike(ABC):
    @abstractmethod
    def as_tensor(self, extra_data: Dict) -> Tensor:
        pass


class State(StateLike):
    def __init__(
        self,
        time_point: float,
        window_skip: float,
        window_length: float,
        observations: Sequence,
        observable_features: Sequence,
    ):
        self.time_point = time_point
        self.cur_window_skip = window_skip
        self.cur_window_length = window_length
        self.observations = observations
        # Not all features in flow_sesh are to be observed
        self.observable_features = observable_features

    # TODO: implement as we find it necessary
    def as_tensor(self, extra_data: Dict) -> Tensor:
        raise NotImplementedError


class Action(np.ndarray):
    """
    Custom ndarray to enforce universal dimensionality and provide specific properties.
    """

    def __new__(cls, data, *args, **kwargs):
        # Convert data to np.ndarray if it's not already
        if isinstance(data, list):
            data = np.array(data)
            assert data.shape[1] == 2, "Data should have exactly two columns"
        elif isinstance(data, np.ndarray):
            assert data.shape[1] == 2, "Data should have exactly two columns"
        else:
            raise TypeError("Data should be a list or np.ndarray")
        # Create the ndarray instance
        obj = np.asarray(data).view(cls)
        return obj

    @property
    def winlength_delta(self):
        return self[:, 0]

    @property
    def winskip_delta(self):
        return self[:, 1]


# class Action(torch.Tensor):
#     """
#     Mostly to enforce unviversal dimensionality.
#     """
#
#     def __new__(cls, data, *args, **kwargs):
#         if isinstance(data, list):
#             assert len(data[0]) == 2, "Data should have exactly two columns"
#         elif isinstance(data, torch.Tensor):
#             assert data.shape[1] == 2, "Data should have exactly two columns"
#         return super().__new__(cls, data, *args, **kwargs)
#
#     @property
#     def winlength_delta(self):
#         return self[:, 0]
#
#     @property
#     def winskip_delta(self):
#         return self[:, 1]
