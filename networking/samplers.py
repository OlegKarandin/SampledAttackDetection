import datetime
from pathlib import Path
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from networking.common_lingo import Attack
from sampleddetection.readers import AbstractTimeSeriesReader
from sampleddetection.samplers import DynamicWindowSampler, binary_search
from sampleddetection.utils import setup_logger


class WeightedSampler(DynamicWindowSampler):
    """
    Assumes that we are dealing with unequally distributes labels
    Expects its timeseries_rdr to contain weights for different time windows
    """

    SHOULD_CACHE = True

    def __init__(
        self,
        timeseries_rdr: AbstractTimeSeriesReader,
        sampling_budget: int,
        num_bins: int,
        labels: Sequence,
        lowest_resolution: float = 1e-6,
    ):
        self.sampling_regions = []
        self.sr_weights = []
        self.timeseries_rdr = timeseries_rdr
        self.logger = setup_logger(__class__.__name__)
        self.labels = labels
        self.logger.info(
            "Going to generate the weights for time windows. This will take a while"
        )
        reg_results = self._generate_regions(num_bins)
        self.bins_times, self.bins_labels, self.perbin_weight = reg_results
        self.bins_labels = self.bins_labels.astype(np.int8)
        self.logger.info("Calculated weights.")

        # Save bin histogram
        self.bin_counts = np.zeros_like(self.perbin_weight)

        super().__init__(
            timeseries_rdr,
            sampling_budget,
            lowest_resolution,
        )

    def _generate_regions(
        self, num_bins: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_dir = ".cache_dir/bins.npy"
        exists_dir = Path(cache_dir).exists()
        if self.SHOULD_CACHE and exists_dir == True:
            # Load it
            self.logger.info("Loading the bins from cache")
            ret_info = np.load(cache_dir)
            return ret_info[0], ret_info[1].astype(np.int8), ret_info[2]

        # Calculate all divisions
        fin_time = self.timeseries_rdr.fin_time
        init_time = self.timeseries_rdr.init_time
        ret_info = np.linspace(init_time, fin_time, num_bins)
        lidx = 0
        num_samples_g = 1000

        # labels_map = {
        #     Attack.BENIGN: Attack.BENIGN.value,
        #     Attack.HULK: Attack.HULK.value,
        #     Attack.GOLDENEYE: Attack.GOLDENEYE.value,
        #     Attack.SLOWLORIS: Attack.SLOWLORIS.value,
        #     Attack.SLOWHTTPTEST: Attack.SLOWHTTPTEST.value,
        # }
        labels_map = [l.value for l in self.labels]

        # Have to figure out how to better add the left bound
        bins_times = [self.timeseries_rdr.init_time]
        bins_labels = [Attack.BENIGN.value]
        for i, b in enumerate(ret_info):
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

            lidx = ridx
            bins_labels.append(label)

        # Distribute sampling weights
        bl_np = np.array(bins_labels, dtype=np.int8)
        uni_weight = 1 / len(labels_map)
        perunit_weigth = np.zeros_like(bl_np, dtype=np.float32)
        for l in labels_map:
            idxs = np.where(bl_np == l)[0]
            perunit_weigth[idxs] = uni_weight / len(idxs)

        # And lets plot it as a histogram just for funsies
        # bt_np = np.array(bins_times)
        # bl_np = np.array(bins_labels, dtype=np.int8)
        # bl_idxs = [bl_np[np.where(bl_np == l)] for l in labels_map.values()]
        # bts = [bt_np[np.where(bl_np == l)] for l in labels_map.values()]
        #
        # fig = plt.figure(figsize=(8, 19))
        # plt.tight_layout()
        # for i, l in enumerate(labels_map.keys()):
        #     print(f"Size of bts[{l}] is {len(bts[i])}")
        #     plt.scatter(bts[i], np.full_like(bts[i], 1), label=ATTACK_TO_STRING[l])
        # plt.legend()
        # plt.show()
        pdb.set_trace()
        if self.SHOULD_CACHE:
            # Make parent
            Path("./.cache_dir/").mkdir(parents=True, exist_ok=True)
            np.save(cache_dir, (bins_times, bins_labels, perunit_weigth))
        return np.array(bins_times, dtype=np.float32), bl_np, perunit_weigth

    def sample(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
        first_sample: bool = False,
    ) -> Sequence[Any]:
        # We first make sure that this is the first sample
        # Chose from one of the bins
        norm_probs = self.perbin_weight[1:] / self.perbin_weight[1:].sum()
        choice_idx = np.random.choice(
            range(len(self.bins_times) - 1), 1, p=norm_probs
        ).item()
        ltime = self.bins_times[choice_idx]
        rtime = self.bins_times[choice_idx + 1]

        # If we are on our first sample we are expected to chose it ourselves
        adj_starting_time = starting_time
        if first_sample == True:
            human_readable = datetime.datetime.fromtimestamp(starting_time)
            adj_starting_time = np.random.uniform(ltime, rtime, 1).item()

        human_readable = datetime.datetime.fromtimestamp(adj_starting_time)

        return super().sample(
            adj_starting_time, window_skip, window_length, initial_precise
        )

    def sample_debug(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
        first_sample: bool = False,
    ) -> Sequence[Any]:
        # We first make sure that this is the first sample
        # Chose from one of the bins
        norm_probs = self.perbin_weight[1:] / self.perbin_weight[1:].sum()
        choice_idx = int(
            np.random.choice(range(len(self.bins_times) - 1), 1, p=norm_probs).item()
        )
        attackid_to_label = {a.value: a for a in self.labels}
        # print(
        #     f"We chosen bin {choice_idx} corresponding to time {self.bins_times[choice_idx]}({to_canadian(self.bins_times[choice_idx])}) and label {attackid_to_label[self.bins_labels[choice_idx]]}"
        # )
        ltime = self.bins_times[choice_idx]
        rtime = self.bins_times[choice_idx + 1]

        # If we are on our first sample we are expected to chose it ourselves
        adj_starting_time = starting_time
        if first_sample == True:
            adj_starting_time = np.random.uniform(ltime, rtime, 1).item()

        actual_sample, debug_sample = super().sample_debug(
            adj_starting_time, window_skip, window_length, initial_precise
        )
        print(f"Chose time {adj_starting_time} from {ltime} to {rtime}")
        print(
            f"With amount of samples {len(actual_sample)} and debug sample {len(debug_sample)}"
        )
        return actual_sample, debug_sample
