from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from time import time
from networking.common_lingo import ATTACK_TO_STRING, STRING_TO_ATTACKS, Attack
from networking.datastructures.flowsession import SampledFlowSession
from networking.datastructures.packet_like import CSVPacket, PacketLike
from sampleddetection.datastructures import CSVSample, SampleLike
from sampleddetection.samplers import FeatureFactory, SampleFactory
from sampleddetection.utils import setup_logger


# TOREM: this just add extra complexity an AbstractTimeSeriesReader can take
class NetworkSampleFactory(SampleFactory[CSVSample]):

    def __init__(self):
        pass

    # Yeah this is kind of ugly
    def make_sample(self, raw_sample: CSVSample) -> SampleLike:
        """
        Returns
        ---------
        - sample (CSVPacket): Will always return a CSVPacket
        """
        # TODO: Maybe generalized
        assert isinstance(
            raw_sample, CSVSample
        ), f"NetworkSampleFactory expects to receive a `CSVPacket`. Instead received {type(raw_sample)}"
        packet_like = CSVPacket(raw_sample.item)
        return packet_like


class NetworkFeatureFactory(FeatureFactory[PacketLike]):

    def __init__(self, observable_features: Sequence[str], labels: Sequence[Attack]):
        self.logger = setup_logger(__class__.__name__)
        self.observable_features: Set[str] = set(observable_features)
        self.expected_labels = (
            list(labels)
            + [
                Attack.BENIGN,
            ]
            if Attack.BENIGN not in labels
            else labels
        )
        self.logger.info(f"Expected labels are {self.expected_labels}")
        self.labels_to_idx: Dict[Attack, int] = {k: i for i, k in enumerate(labels)}
        self.strings_to_idx: Dict[str, int] = {
            ATTACK_TO_STRING[k]: i for i, k in enumerate(self.expected_labels)
        }
        # For t

    def make_feature_and_label(
        self, raw_sample_list: Sequence[PacketLike]
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.logger.debug(
            f"Constructing features and labels for {len(raw_sample_list)} packets"
        )
        flowsession = SampledFlowSession()
        flowsession.reset()

        for raw_sample in raw_sample_list:
            curt = time()
            flowsession.on_packet_received(raw_sample)
            self.logger.debug(
                f"Time to add packet (from make_feature_and_label) is {time() - curt}"
            )

        # Once all the flow is retrieved we create an array-like
        self.logger.debug(
            f"Now calculating all the data for flow session with {len(flowsession.flows)}"
        )
        data: Dict[Tuple, Dict] = flowsession.get_data()

        raw_features = []
        raw_labels = []
        # TODO:: Probably check for empty returns
        self.logger.debug(f"We have obtaine {len(data)} flows to get info from.")
        for flow_key, feat_dict in data.items():
            # Avoid packets we dont care for
            label_str = feat_dict["label"]
            label_enum = STRING_TO_ATTACKS[label_str]
            if label_enum not in self.expected_labels:
                self.logger.debug(f"Skipping on label_id {label_enum}")
                continue
            # Fetch all features and labels as specified by keys in self.observable_features
            raw_features.append(self._get_flow_feats(feat_dict))
            self.logger.debug(
                f"(flow_key:{flow_key}) Going through label string {label_str}"
                f" with id {self.strings_to_idx[label_str]}"
            )
            raw_labels.append(self.strings_to_idx[label_str])

        self.logger.debug(f"Raw labels {raw_labels}")

        # CHECK: if this is a valid way of giving it a default state
        if len(raw_labels) == 0 and len(raw_features) == 0:
            arraylike_features = np.array(
                [[0 for _ in range(len(self.observable_features))]]
            )
            arraylike_labels = np.array([[0]], dtype=np.int16)
        else:
            arraylike_features = np.array(raw_features)
            arraylike_labels = np.array(raw_labels, dtype=np.int16)

        self.logger.debug(
            f"Our features from a sample looks like:\n{arraylike_features}"
        )
        return arraylike_features, arraylike_labels

    def get_feature_strlist(self):
        return self.observable_features

    def _get_flow_feats(self, feat_dict: Dict[str, float]) -> List:
        sample_features = []
        for feat_name, v in feat_dict.items():
            if feat_name in self.observable_features:
                if isinstance(v, str):  # i.e if is label
                    if v not in list(ATTACK_TO_STRING.values()):
                        raise ValueError(
                            f"Received string does not correspond to label."
                        )
                    else:
                        v = self.strings_to_idx[v]
                sample_features.append(float(v))
        return sample_features
