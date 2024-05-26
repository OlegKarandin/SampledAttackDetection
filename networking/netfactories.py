from enum import Enum
from typing import Dict, Sequence, Set, Tuple

import numpy as np

from networking.common_lingo import ATTACK_TO_STRING, STRING_TO_ATTACKS, Attack
from networking.datastructures.flowsession import SampledFlowSession
from networking.datastructures.packet_like import CSVPacket, PacketLike
from sampleddetection.datastructures import CSVSample
from sampleddetection.samplers import FeatureFactory, SampleFactory
from sampleddetection.utils import setup_logger


class NetworkSampleFactory(SampleFactory):

    def __init__(self):
        pass

    # Yeah this is kind of ugly
    def make_sample(self, raw_sample: CSVSample) -> CSVPacket:

        # TODO: Maybe generalized
        assert isinstance(
            raw_sample, CSVSample
        ), f"NetworkSampleFactory expects to receive a `CSVPacket`. Instead received {type(raw_sample)}"
        packet_like = CSVPacket(raw_sample.item)
        return packet_like


class NetworkFeatureFactory(FeatureFactory):

    def __init__(self, observable_features: Sequence[str], labels: Sequence[Attack]):
        self.logger = setup_logger(__class__.__name__)
        self.observable_features: Sequence[str] = observable_features
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

    def make_feature(
        self, raw_sample_list: Sequence[PacketLike]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Clean the previous flow session
        flowsession = SampledFlowSession()
        flowsession.reset()
        for raw_sample in raw_sample_list:
            flowsession.on_packet_received(raw_sample)

        # Use FlowSession to get the data and get the feature

        # Once all the flow is retrieved we create an array-like
        # We can use to do inference
        data = flowsession.get_data()

        raw_features = []
        # TODO: Clean
        for flow_key, feat_dict in data.items():
            # Fetch all features as specified by keys in self.observable_features
            raw_features.append(self._get_flow_feats(feat_dict))
            # Also fetch the labels for each feature.
            raw_labels.append(self.get_feature_strlist)

        arraylike_features = np.array(raw_features)

        self.logger.debug(
            f"Our features from a sample looks like:\n{arraylike_features}"
        )
        return arraylike_features

    def get_feature_strlist(self):
        return self.observable_features

    def _get_flow_feats(self, feat_dict):
        sample_features = []
        for feat_name, v in feat_dict.items():
            if feat_name in self.observable_features:
                if isinstance(v, str):
                    if v not in list(ATTACK_TO_STRING.values()):
                        raise ValueError(
                            f"Received string does not correspond to label."
                        )
                    else:
                        v = self.strings_to_idx[v]
                sample_features.append(float(v))
        return sample_features
