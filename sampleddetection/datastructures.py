from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
from common_lingo import ArrayLike
from torch import Tensor

from ..datastructures.flowsession import SampledFlowSession


# Just define by properties one expects it to have
class State(ABC):
    @abstractmethod
    def as_tensor(self, extra_data: Dict) -> Tensor:
        pass


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
