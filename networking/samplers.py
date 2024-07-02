from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from networking.common_lingo import ATTACK_TO_STRING, STRING_TO_ATTACKS, Attack
from sampleddetection.readers import AbstractTimeSeriesReader
from sampleddetection.samplers import DynamicWindowSampler, SampleFactory, binary_search


class WeightedSampler(DynamicWindowSampler):
    """
    Assumes that we are dealing with unequally distributes labels
    Expects its timeseries_rdr to contain weights for different time windows
    """

    def __init__(
        self,
        timeseries_rdr: AbstractTimeSeriesReader,
        # TOREM: Not really being used
        sampling_budget: int,
        num_bins: int,
        lowest_resolution: float = 1e-6,
    ):
        self.sampling_regions = []
        self.sr_weights = []
        self._generate_regions(num_bins)
        super().__init__(
            timeseries_rdr,
            sampling_budget,
            lowest_resolution,
        )

    def _generate_regions(self, num_bins: int):
        # Calculate all divisions
        fin_time = self.timeseries_rdr.fin_time
        init_time = self.timeseries_rdr.init_time
        tot_time = fin_time - init_time
        bins = np.linspace(init_time, fin_time, num_bins)
        lidx = 0
        num_samples_g = 1000

        labels_map = {
            Attack.BENIGN: Attack.BENIGN.value,
            Attack.HULK: Attack.HULK.value,
            Attack.GOLDENEYE: Attack.GOLDENEYE.value,
            Attack.SLOWLORIS: Attack.SLOWLORIS.value,
            Attack.SLOWHTTPTEST: Attack.SLOWHTTPTEST.value,
        }

        bins_times = []
        bins_labels = []
        for i, b in enumerate(bins):
            if i == 0:
                continue
            # Find the timestamp
            ridx = binary_search(self.timeseries_rdr, b)
            bins_times.append(b)

            # Just grab the first num_samples packets here and uset them to decide
            num_samples = min(num_samples_g, ridx - lidx)
            packs = self.timeseries_rdr[lidx : lidx + num_samples]
            packs_lbls = np.array([p.label.value for p in packs], dtype=np.int8)
            label = Attack.BENIGN.value  # By default
            if sum(packs_lbls == Attack.BENIGN.value) != num_samples:
                label = packs_lbls[packs_lbls != Attack.BENIGN.value][0]
                print("|", end="")

            lidx = ridx
            bins_labels.append(label)

        # And lets plot it as a histogram just for funsies
        bt_np = np.array(bins_times)
        bl_np = np.array(bins_labels, dtype=np.int8)
        bl_idxs = [bl_np[np.where(bl_np == l)] for l in labels_map.values()]
        bts = [bt_np[np.where(bl_np == l)] for l in labels_map.values()]

        fig = plt.figure(figsize=(8, 19))
        plt.tight_layout()
        for i, l in enumerate(labels_map.keys()):
            print(f"Size of bts[{l}] is {len(bts[i])}")
            plt.scatter(bts[i], np.full_like(bts[i], 1), label=ATTACK_TO_STRING[l])
        plt.legend()
        plt.show()
        return bins_times, bins_labels

    def sample(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
        **kwargs,
    ) -> Sequence[Any]:
        # We first make sure that this is the first sample
        assert "fist_sample" in kwargs

        return super().sample(
            starting_time, window_skip, window_length, initial_precise
        )
