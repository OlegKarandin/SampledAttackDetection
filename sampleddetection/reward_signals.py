from abc import ABC, abstractmethod
from typing import Any, List

import joblib as joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


class RandForRewardCalculator(RewardCalculatorLike):
    TRANS_DICT = {
        "fwd_pkt_len_max": "Fwd Packet Length Max",
        "fwd_pkt_len_min": "Fwd Packet Length Min",
        "fwd_pkt_len_mean": "Fwd Packet Length Mean",
        "bwd_pkt_len_max": "Bwd Packet Length Max",
        "bwd_pkt_len_min": "Bwd Packet Length Min",
        "bwd_pkt_len_mean": "Bwd Packet Length Mean",
        "flow_byts_s": "Flow Bytes/s",
        "flow_pkts_s": "Flow Packets/s",
        "flow_iat_mean": "Flow IAT Mean",
        "flow_iat_max": "Flow IAT Max",
        "flow_iat_min": "Flow IAT Min",
        "fwd_iat_mean": "Fwd IAT Mean",
        "fwd_iat_max": "Fwd IAT Max",
        "fwd_iat_min": "Fwd IAT Min",
        "bwd_iat_max": "Bwd IAT Mean",
        "bwd_iat_min": "Bwd IAT Max",
        "bwd_iat_mean": "Bwd IAT Min",
        "pkt_len_min": "Min Packet Length",
        "pkt_len_max": "Max Packet Length",
        "pkt_len_mean": "Packet Length Mean",
    }

    def __init__(self, model_path: str):
        # Load the model
        self.model = joblib.load(model_path)
        self.logger = setup_logger(__class__.__name__)

    def calculate(self, **observations) -> float:
        assert isinstance(observations["ground_truths"], np.ndarray)
        assert isinstance(observations["features"], np.ndarray)

        grounded_truths = torch.LongTensor(observations["ground_truths"])
        features = torch.tensor(observations["features"], dtype=torch.float)
        features_df = pd.DataFrame(
            observations["features"], columns=list(self.TRANS_DICT.values())
        )
        if grounded_truths.sum() + features.sum() == 0:
            return -10  # Some default very low value that should mean bad reward
        predictions = self.model.predict_proba(features_df)
        self.logger.debug(f"Predictions are looking like {predictions}")
        predictions = torch.from_numpy(predictions)
        self.logger.debug(
            f"Tensored Predictions are looking like {predictions} with ground_truths : {grounded_truths}"
        )
        losses = F.nll_loss(predictions, grounded_truths)
        self.logger.debug(f"Losses theselves {losses}")

        final_loss = -1 * losses.mean().item()
        self.logger.debug(f"Final Losses theselves {final_loss}")
        return final_loss


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
            f"Criterion: \n\tinput:{F.softmax(predictions)}\n\ground truth:{grounded_truths}"
        )
        losses = self.criterion(predictions, grounded_truths)
        self.logger.debug(f"Resulting losses {losses}")

        loss_mean = losses.mean() * (-1)
        self.logger.debug(f"Mening mean reward is {loss_mean}")
        # Reinforcement Learning need not gradients.

        # Do SGD over the loss so we can learn to better classify.
        # self.estimation_signal.zero_grad()
        # loss_mean.backward()
        # self.optim.step()
        # CHECK: that we are doing sgd well here.

        return loss_mean.item()

    def learn(self, features: np.ndarray, ground_truths: np.ndarray):
        pass  # TODO:


# TODO: One might imagine a milar reward calculator for random forests/trees
