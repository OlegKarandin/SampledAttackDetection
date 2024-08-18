from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

from networking.common_lingo import Attack
from sampleddetection.readers import AbstractTimeSeriesReader
from sampleddetection.samplers import BasicSampler , binary_search
from tqdm import tqdm
from sampleddetection.utils import setup_logger


class WeightedSampler(BasicSampler):
    """
    Assumes that we are dealing with unequally distributed labels
    Expects its timeseries_rdr to contain weights for different time windows
    """

    SHOULD_CACHE = True

    def __init__(
        self,
        timeseries_rdr: AbstractTimeSeriesReader,
        sampling_budget: int,
        num_bins: int,
        labels: Sequence,
        binary_labels: bool = False,
        lowest_resolution: float = 1e-6,
    ):
        super().__init__(timeseries_rdr, sampling_budget,  lowest_resolution)
        self.labels = labels
        self.logger.info(
            "Generating the weights for sampling at time windows. This will take a while"
        )

        (
          self.bins_times,
          self.bins_labels,
          self.perbin_weight
        ) = self._generate_regions(num_bins, binary_labels)
    def _generate_regions(
        self, num_bins: int, binary_labels: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Will generate the bins and weights for the weighted sampler
        Args:
            num_bins: The amount of bins to generate
            binary_labels: Whether or not to use binary labels
        Returns:
            bins_times: The times of the bins
            bins_labels: The labels of the bins
            perunit_weight: The weight of each bin
        """
        cache_dir = ".cache_dir/bins.npy"
        exists_dir = Path(cache_dir).exists()
        if self.SHOULD_CACHE and exists_dir == True:
            # Load it
            self.logger.info("Loading the bins from cache")
            ret_info = np.load(cache_dir)
            return ret_info[0], ret_info[1].astype(np.int8), ret_info[2]

        # Preparete for the loop
        fin_time = self.timeseries_rdr.fin_time
        init_time = self.timeseries_rdr.init_time
        ret_info = np.linspace(init_time, fin_time, num_bins)
        num_samples_g = 1000
        if binary_labels:
            labels_dict = {
                Attack.BENIGN: Attack.BENIGN.value,
                Attack.GENERAL: Attack.GENERAL.value,
            }
        else:
            labels_dict = {
                Attack.BENIGN: Attack.BENIGN.value,
                Attack.HULK: Attack.HULK.value,
                Attack.GOLDENEYE: Attack.GOLDENEYE.value,
                Attack.SLOWLORIS: Attack.SLOWLORIS.value,
                Attack.SLOWHTTPTEST: Attack.SLOWHTTPTEST.value,
            }
        attackAvail_enumVals = (
            [l.value for l in self.labels]
            if not binary_labels
            else [Attack.BENIGN.value, Attack.GENERAL.value]
        )

        # TODO: Have to figure out how to better add the left bound
        lidx = 0
        bins_times = [self.timeseries_rdr.init_time]
        bins_enumVals = [Attack.BENIGN.value]
        for i, b in enumerate(tqdm(ret_info[:-1], desc="Generating bins")):
            if i == 0:
                continue
            # Find the timestamp
            ridx = binary_search(self.timeseries_rdr, b)
            bins_times.append(b)

            # Decide label
            # Just grab the first num_samples packets here and use them to decide
            num_samples = min(num_samples_g, ridx - lidx)
            packs = self.timeseries_rdr[lidx : lidx + num_samples]
            packs_lbls = np.array([p.label.value for p in packs], dtype=np.int8)
            label = Attack.BENIGN.value  # By default
            if sum(packs_lbls == Attack.BENIGN.value) != num_samples:
                label = (
                    packs_lbls[packs_lbls != Attack.BENIGN.value][0]
                    if not binary_labels
                    else Attack.GENERAL.value
                )

            lidx = ridx
            bins_enumVals.append(label)

        # Distribute sampling weights
        bin_labels_np = np.array(bins_enumVals, dtype=np.int8)
        unit_weight = 1 / len(attackAvail_enumVals)
        perunit_weight = np.zeros_like(bin_labels_np, dtype=np.float32)
        for l in attackAvail_enumVals:
            idxs = np.where(bin_labels_np == l)[0]
            if len(idxs) == 0:
                continue
            perunit_weight[idxs] = unit_weight / len(idxs)
        bin_times_np = np.array(bins_times, dtype=np.float32)

        # # For debugging
        # labelGroup_bin_times = [bin_times_np[np.where(bin_labels_np == l)] for l in labels_dict.values()]
        # plt.figure(figsize=(8, 19))
        # plt.tight_layout()
        # for i, l in enumerate(labels_dict.keys()):
        #     print(f"Size of bts[{l}] is {len(labelGroup_bin_times[i])}")
        #     plt.scatter(labelGroup_bin_times[i], np.full_like(labelGroup_bin_times[i], 1), label=ATTACK_TO_STRING[l])
        # plt.legend()
        # plt.show()

        if self.SHOULD_CACHE:
            # Make parent
            Path("./.cache_dir/").mkdir(parents=True, exist_ok=True)
            np.save(cache_dir, (bin_times_np, bin_labels_np, perunit_weight))
        return bin_times_np, bin_labels_np, perunit_weight


    def sample_starting_point(self, margin: float = 0.95) -> float:
        """
        Will take weights calcualted in _generate_regions and return a random point between the first and last
        but sampled according to the weights
        """
        # Ensure that the weights are normalized,
        assert np.isclose(np.sum(self.perbin_weight), 1), "Weights are not normalized"
        choice_idx = np.random.choice(
            # -2 is just so that we can calculate bin time in next line
            range(len(self.bins_times)), 1, p=self.perbin_weight
        ).item()
        bin_time = self.bins_times[choice_idx]
        # We will sample within it now
        sample_time = np.random.uniform(bin_time, self.bins_times[choice_idx + 1])
        assert isinstance(sample_time,float)
        return sample_time

        
    def sample(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
    ) -> Sequence[Any]:

        return super().sample(
            starting_time, window_skip, window_length, initial_precise
        )

    # CHECK: Havent refactored it in a while compared to surrounding code
    # Maybe there are changes pending here
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
