from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import torch
from torch import nn, optim

from sampleddetection.utils import setup_logger


class RewardCalculatorLike(ABC):
    """
    One can imagine that these calculators will take take observations and return an evaluation metric.
    These will likely always come tightly coupled with some downstream task
    """

    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Will return a performance metric based on parameters passed.
        Paraneters passed will depend a lot on the downstream argument so Ill leave them as a dictionary

        Returns
        ---------
        - float: Performance metric
        """

        pass


"""
Some basic reward calculators follow
"""


class DNN_RewardCalculator(RewardCalculatorLike):

    def __init__(self, estimation_signal: nn.Module):

        # self.groundtruth_ds = groundtruth_ds
        self.estimation_signal = estimation_signal  # probably an DNN
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.SGD(estimation_signal.parameters(), lr=1e-3)
        self.logger = setup_logger(__class__.__name__)
        # Set up optimizers here if we want to simultaneously train whatever network we sned

    def calculate(self, **observations) -> float:
        # corresponding_idxs: List[int] = kwargs["corresponding_idxs"]
        assert isinstance(observations["ground_truths"], np.ndarray)
        assert isinstance(observations["features"], np.ndarray)

        grounded_truths = torch.LongTensor(observations["ground_truths"])
        features = torch.tensor(observations["features"], dtype=torch.float)
        # CHECK: Experimental. Figure out a better method: <Begin>
        # If we have no samples at all. (We have written this as all 0s for now)
        if grounded_truths.sum() + features.sum() == 0:
            return -10  # Some default very low value that should mean bad reward

        # CHECK: Experimental. Figure out a better method: <End>

        # Convert them to tensors
        # predictions: List[int] = kwargs["predictions"]
        predictions = self.estimation_signal(features)

        # Observe what the actual label is
        # retrieved_truths = self.groundtruth_ds.retrieve_truth()
        # Some categorigal loss here
        # losses = self.criterion(predictions, retrieved_truths)
        self.logger.debug(
            f"Criterion: \n\tinput:{predictions}\n\touttput:{grounded_truths}"
        )
        losses = self.criterion(predictions, grounded_truths)
        self.logger.debug(f"Resulting losses {losses}")

        loss_mean = losses.mean() * (-1)
        loss_mean = loss_mean * -1  # Convert to reward
        # Reinforcement Learning need not gradients.

        # Do SGD over the loss so we can learn to better classify.
        self.estimation_signal.zero_grad()
        loss_mean.backward()
        self.optim.step()
        # CHECK: that we are doing sgd well here.

        return loss_mean.item()


# TODO: One might imagine a milar reward calculator for random forests/trees
