from abc import ABC, abstractmethod
from typing import Dict, NamedTuple

import numpy as np
import pandas as pd
from torch import Tensor


# Just define by properties one expects it to have
# CHECK: if we actually want to use this approach
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
        # First axis will index individual samples
        observations: np.ndarray,
        # observations: Sequence,
        # observable_features: Sequence[Sample],
    ):
        self.time_point = time_point
        self.window_skip = window_skip
        self.window_length = window_length
        self.observations = observations
        # Not all features in flow_sesh are to be observed
        # self.observable_features = observable_features

    # TODO: implement as we find it necessary
    def as_tensor(self, extra_data: Dict) -> Tensor:
        raise NotImplementedError

    def as_numpy(self, extra_delta: Dict) -> np.ndarray:
        raise NotImplementedError

    def as_arraylike(self, extra_data: Dict) -> np.ndarray:
        raise NotImplementedError


# class Action(np.ndarray):
#     """
#     Custom ndarray to enforce universal dimensionality and provide specific properties.
#     """
#
#     def __new__(cls, data, *args, **kwargs):
#         # Convert data to np.ndarray if it's not already
#         if isinstance(data, list):
#             data = np.array(data)
#             assert data.shape[1] == 2, "Data should have exactly two columns"
#         elif isinstance(data, np.ndarray):
#             assert data.shape[1] == 2, "Data should have exactly two columns"
#         else:
#             raise TypeError("Data should be a list or np.ndarray")
#         # Create the ndarray instance
#         obj = np.asarray(data).view(cls)
#         return obj
#
#     @property
#     def winlength_delta(self):
#         return self[:, 0]
#
#     @property
#     def winskip_delta(self):
#         return self[:, 1]


class Action(NamedTuple):
    winskip_delta: int
    winlen_delta: int


class SampleLike(ABC):
    """
    Define a few attributes that samples must have
    """

    @property
    @abstractmethod
    def time(self) -> float:
        pass


# TODO: A bit loose for my liking
class Sample(SampleLike):
    def __init__(self, item):
        self.item = item

    @property
    def time(self) -> float:
        """The time property."""
        return self.item.time


class CSVSample(SampleLike):
    def __init__(self, row: pd.Series):
        self.item = row

    @property
    def time(self) -> float:
        """The time property."""
        return self.item["timestamp"]

    def __str__(self) -> str:
        str = f"CSVSample: {self.item}"
        return str


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
