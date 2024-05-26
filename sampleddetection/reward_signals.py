from abc import ABC, abstractmethod
from typing import Any, List

from torch import nn, optim


class RewardCalculatorLike(ABC):
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        pass


"""
Some basic reward calculators follow
"""


class DNN_RewardCalculator(RewardCalculatorLike):
    """
    One can imagine that these calculators will take their estimation and thei ground truth
    """

    def __init__(self, groundtruth_ds: Any, estimation_signal: nn.Module):

        self.groundtruth_ds = groundtruth_ds
        self.estimation_signal = estimation_signal  # probably an DNN
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.SGD(estimation_signal.parameters(), lr=1e-3)
        # Set up optimizers here if we want to simultaneously train whatever network we sned

    def calculate(self, **kwargs) -> float:
        corresponding_idxs: List[int] = kwargs["corresponding_idxs"]
        predictions: List[int] = kwargs["predictions"]

        # Observe what the actual label is
        retrieved_truths = self.groundtruth_ds.retrieve_truth()
        # Some categorigal loss here
        losses = self.criterion(predictions, retrieved_truths)

        loss_mean = losses.mean()
        # Reinforcement Learning need not gradients.

        return losses.item()
