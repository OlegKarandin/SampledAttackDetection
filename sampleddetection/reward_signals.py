from abc import ABC, abstractmethod
from typing import Dict, Tuple

import joblib as joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn, optim
from collections import OrderedDict

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
        Parameters passed will depend a lot on the downstream argument so Ill leave them as a dictionary

        Returns
        ---------
        - float: Performance metric
        - dict: For extra metrics for debugging
        """

        pass

    @ abstractmethod
    def calculate_eval(self, **kwargs) -> Tuple[float, Dict]:
        """
        Similar as above but for validation and any extra metrics
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

    def __init__(self, model_path: str, binary_labels: bool = False):
        # Load the model
        self.model = joblib.load(model_path)
        self.logger = setup_logger(__class__.__name__)
        self.binary_labels = binary_labels

    def calculate(self, **observations) -> float:
        assert isinstance(observations["ground_truths"], np.ndarray)
        assert isinstance(observations["features"], np.ndarray)

        grounded_truths = torch.LongTensor(observations["ground_truths"])
        features = torch.tensor(observations["features"], dtype=torch.float)

        # So we can feed it into the RandForest Model
        self.logger.debug("Checking if we have any samples")
        features_df = pd.DataFrame(
            observations["features"],
            columns=list(self.TRANS_DICT.values()) # type:ignore
        )
        # Check if we didnt get any samples
        if grounded_truths.sum() + features.sum() == 0:
            return -10  # Some default very low value that should mean bad reward
        
        ### Run the model
        predictions = self.model.predict_proba(features_df)
        predictions = torch.from_numpy(predictions)

        self.logger.debug(f"Predictions are looking like {predictions}")

        self.logger.debug(f"About to calculate the loss. Binary labels are {self.binary_labels}")
        final_loss = 0
        if self.binary_labels:
            final_loss = self._binary_losses(grounded_truths, predictions)
        else:
            final_loss = self._multinary_loses(grounded_truths, predictions)
        self.logger.debug(f"Loss returned by reward calculator is: {final_loss}")
        
        return final_loss

    # TODO: Finish this
    def _binary_losses(self, grounded_truths, predictions) -> float:
        loss = F.nll_loss(predictions, grounded_truths)
        loss = -1  * loss.mean().item()

        return loss

    def _multinary_loses(self, grounded_truths, predictions) -> float:
        ### Calculate the Loss
        losses = F.nll_loss(predictions, grounded_truths)
        final_loss = -1 * losses.mean().item()
        return final_loss

    def calculate_eval(self, **observations) -> Tuple[float, Dict]:
        assert isinstance(observations["ground_truths"], np.ndarray)
        assert isinstance(observations["features"], np.ndarray)

        # Calculate the reward
        grounded_truths = torch.LongTensor(observations["ground_truths"])
        features = torch.tensor(observations["features"], dtype=torch.float)
        features_df = pd.DataFrame(
            observations["features"],
            columns=list(self.TRANS_DICT.values()) # type:ignore
        )
        #TODO: Fix this scalar to something that makes sense
        # This is the only way so far to check for no samples
        if grounded_truths.sum() + features.sum() == 0:
            # Some default very low value that should mean bad reward
            return -10, {"loss": -10}

        predictions = self.model.predict_proba(features_df)
        predictions = torch.from_numpy(predictions)

        if self.binary_labels:
            eval_metrics = self._binary_eval(grounded_truths, predictions)
        else:
            eval_metrics = self._multinary_eval(grounded_truths, predictions)

        # Calculate Confusion Matrix

        return (next(iter(eval_metrics.values())), eval_metrics)

    def _binary_eval(self, grounded_truths, pred_probs) -> OrderedDict:
        # Calculate accuracy
        # CHECK: Whethere ther are actually two probs or not
        self.logger.debug(f"Predictions (of shape {pred_probs.shape}) are looking like {pred_probs}")
        self.logger.debug(f"Whereas ground truths (of shape {grounded_truths.shape}) are looking like {grounded_truths}")
        preds = torch.argmax(pred_probs, dim=1)
        acc = torch.sum(preds == grounded_truths).item() / len(preds)
        self.logger.debug(f"Accuracy is {acc}")

        # Calculate AUC
        # auc = roc_auc_score(grounded_truths, pred_probs[:, 1],multi_class = "ovr")
        
        # Calculate Type I Error
        tp = torch.sum(preds[preds == 1] == grounded_truths[preds == 1]).item()
        fp = torch.sum(preds[preds == 1] != grounded_truths[preds == 1]).item()
        tn = torch.sum(preds[preds == 0] == grounded_truths[preds == 0]).item()
        fn = torch.sum(preds[preds == 0] != grounded_truths[preds == 0]).item()

        # type_i_error = tp / (tp + fn)
        tot_neg = fp + tn
        tot_pos = fn + tp
        type_i_error = fp / (tot_neg) if tot_neg != 0 else None
        type_ii_error = fn / (tot_pos) if tot_pos != 0 else None

        self.logger.debug(
            f"Type I Error is {type_i_error}.\n" f"Type II Error is {type_ii_error}"
        )

        losses = F.nll_loss(pred_probs, grounded_truths)
        mean_nll_loss = -1 * losses.mean().item()
        self.logger.debug(f"Mean NLL Loss is {mean_nll_loss}")

        eval_dict = OrderedDict(
            {
                "accuracy": 0,
                "nll_loss": 0,
                "type_i_error": 0,
                "type_ii_error": 0,
            }
        )
        eval_dict = OrderedDict(
            {
                "accuracy": acc,
                "nll_loss": mean_nll_loss,
                "type_i_error": type_i_error,
                "type_ii_error": type_ii_error,
            }
        )

        return eval_dict

    # TODO: This whole function needs some love
    def _multinary_eval(self, grounded_truths, predictions) -> OrderedDict:
        # Calculate accuracy
        preds = torch.argmax(predictions, dim=1)
        acc = torch.sum(preds == grounded_truths).item() / len(preds)
        self.logger.debug(f"Accuracy is {acc}")

        # TODO:
        # Calculate AUC
        # pdb.set_trace()
        # auc = roc_auc_score(grounded_truths, pred_probs[:, 1],multi_class = "ovr")
        # cf_matrix = confusion_matrix(grounded_truths, preds)
        
        # Final Return
        eval_dict = {
            "accuracy": acc,
            # "auc": auc,
        }
        return eval_dict



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
            f"Criterion: \n\tinput:{F.softmax(predictions)}\n"
             "ground truth:{grounded_truths}"
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

