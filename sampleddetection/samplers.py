import logging
import time
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Generic, List, Sequence, Tuple, TypeVar

import numpy as np

from networking.datastructures.packet_like import CSVPacket
from sampleddetection.datastructures import SampleLike
from sampleddetection.readers import AbstractTimeSeriesReader

from .common_lingo import TimeWindow
from .utils import setup_logger


class TSSampler(ABC):
    @abstractmethod
    def sample(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
    ) -> Sequence[Any]:
        pass

    @abstractmethod
    def window_statistics(
        self, starting_time: float, win_skip: float, win_len: float
    ) -> dict:
        pass

    @property
    @abstractmethod
    def init_time(self) -> float:
        pass

    @property
    @abstractmethod
    def fin_time(self) -> float:
        pass


T = TypeVar("T")


class SampleFactory(Generic[T]):

    @abstractmethod
    def make_sample(self, raw_sample: T) -> SampleLike:
        pass


T = TypeVar("T")


class FeatureFactory(ABC, Generic[T]):

    @abstractmethod
    def make_feature_and_label(
        self, raw_sample_list: Sequence[T]
    ) -> Tuple[np.ndarray, np.ndarray]:  # Features and Labels
        pass  # TODO: Figure out what it is that we want to return here.

    @abstractmethod
    def get_feature_strlist(self) -> Sequence[str]:
        pass


class DynamicWindowSampler(TSSampler):
    """
    Sampler Agnostic to Type of data being dealt with.
    """

    def __init__(
        self,
        timeseries_rdr: AbstractTimeSeriesReader,
        # TOREM: Not really being used
        sampling_budget: int,
        lowest_resolution: float = 1e-6,
    ):
        self.lowest_resolution = lowest_resolution
        self.logger = setup_logger(__class__.__name__, logging.DEBUG)
        self.timeseries_rdr = timeseries_rdr
        self.sampling_budget = sampling_budget

        self.max_idx = len(self.timeseries_rdr) - 1

    @property
    def init_time(self):
        # Send back as normal python float (rather than np.float64)
        return float(self.timeseries_rdr.init_time)

    @property
    def fin_time(self):
        # Send back as normal python float (rather than np.float64)
        return float(self.timeseries_rdr.fin_time)

    def sample(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
    ) -> Sequence[Any]:
        # ) -> SampledFlowSession:
        # ) -> pd.DataFrame:
        """
        Will just return a list of samples

        Parameters
        ~~~~~~~~~~
            e  - initial_precise: Whether we shoudl staat precisely at the provided time or at the closest packet to it
        """
        # self.logger.debug(
        #     f"Entering with starting_time {starting_time} (of type {type(starting_time)}), window_skip {window_skip} and window_length {window_length}"
        # )

        _starting_time = starting_time
        _stopping_time = starting_time + window_length

        samples = []
        for s in range(self.sampling_budget):
            idx_firstsamp = binary_search(self.timeseries_rdr, _starting_time)
            idx_lastsamp = binary_search(self.timeseries_rdr, _stopping_time)

            # self.logger.debug(
            #     f"Entering with starting_time {_starting_time} and ending at {_stopping_time}"
            # )
            # self.logger.debug(f"Will sample between {idx_firstsamp} -> {idx_lastsamp}")

            # This call might be IPC so be careful not to abuse it
            cur_samples = self.timeseries_rdr[idx_firstsamp:idx_lastsamp]
            samples += cur_samples

            ### DEBUG:
            # for c in cur_samples:
            #     self.logger.debug(f"\tSampled packet at time {c.time}")

            # self.logger.debug(
            #     f"{s}thn (Win:{_starting_time}->{_stopping_time}) batch contains {len(cur_samples)} samples"
            # )
            _starting_time += window_skip
            _stopping_time = _starting_time + window_length

        return samples

    def sample_debug(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
        first_sample: bool = False,
    ) -> Tuple[Sequence[Any], Sequence[Any]]:
        # ) -> SampledFlowSession:
        # ) -> pd.DataFrame:
        """
        Will just return a list of samples

        Parameters
        ~~~~~~~~~~
            e  - initial_precise: Whether we shoudl staat precisely at the provided time or at the closest packet to it
        """

        _starting_time = starting_time
        _stopping_time = starting_time + window_length

        assert self.sampling_budget > 0, "Sampling budget must be greater than 0"
        samples = []
        for s in range(self.sampling_budget):
            idx_firstsamp = binary_search(self.timeseries_rdr, _starting_time)
            idx_lastsamp = binary_search(self.timeseries_rdr, _stopping_time)

            # This call might be IPC so be careful not to abuse it
            cur_samples = self.timeseries_rdr[idx_firstsamp:idx_lastsamp]
            samples += cur_samples

            _starting_time += window_skip
            _stopping_time = _starting_time + window_length

        _veryfirst_idx = binary_search(self.timeseries_rdr, starting_time)
        _actual_final_stopping_time = _starting_time
        all_dem_samples = self.timeseries_rdr[_veryfirst_idx:idx_lastsamp]

        return samples, all_dem_samples

    # TODO: I doubt we are actually going to be using this
    def window_statistics(
        self, starting_time: float, win_skip: float, win_len: float
    ) -> dict:

        st_time = time.time()
        samples = []
        end_time = starting_time + (win_skip + win_len) * self.sampling_budget
        start_idx = binary_search(self.timeseries_rdr, starting_time)
        end_idx = binary_search(self.timeseries_rdr, end_time)

        self.logger.debug(
            f"At start time {starting_time} we are given a win_skip:{win_skip} and wind_len {win_len}"
        )
        self.logger.debug(
            f"We are pottentially going through {end_idx-start_idx} ({start_idx}->{end_idx}) samples"
        )
        all_samples = self.timeseries_rdr[start_idx:end_idx]

        ### Calculate Statistics
        ## We will bravely assume that we are dealing with `PacketLike` here
        times = np.array([s.time for s in all_samples])
        iats = times[1:] - times[:-1]

        stats = {
            "mean_iat": iats.mean() if len(iats) != 0 else 0,
            "min_iat": iats.min() if len(iats) != 0 else 0,
            "max_iat": iats.max() if len(iats) != 0 else 0,
        }
        self.logger.debug(
            f"window_statiscs took {time.time() - st_time} with {len(samples)} samples between st_time:{st_time} to end_time:{end_time}"
        )
        return stats


class NoReplacementSampler(DynamicWindowSampler):
    def __init__(
        self,
        csvrdr: AbstractTimeSeriesReader,
        sampling_budget: int,
        specific_sample_factory: SampleFactory,
        lowest_resolution: float = 1e-6,
    ):
        super().__init__(csvrdr, sampling_budget, lowest_resolution)
        self.sampled_window_count = 0
        self.sampled_windows: List[TimeWindow] = []

    def sample(
        self,
        starting_time: float,
        window_skip: float,
        window_length: float,
        initial_precise: bool = False,
        # ) -> SampledFlowSession:
    ) -> Sequence[Any]:
        """
        starting_time will be ignored
        """
        foundSampled = True
        local_starting_time = -1
        while foundSampled:
            local_starting_time = np.random.uniform(
                low=self.init_time, high=self.fin_time
            )
            end_time = local_starting_time + window_length

            foundSampled = False  # Start with hope
            # OPTIM:  This will likely get slow very quickly
            for window in self.sampled_windows:
                if (
                    window.start <= local_starting_time
                    and local_starting_time <= window.end
                ) or (window.start <= end_time and end_time <= window.end):
                    foundSampled = True
                    self.logger.warn(
                        f"Found a clash! (#{self.sampled_window_count})"
                        f"Trying window start {local_starting_time} with end_time {end_time}"
                        f"Clashed with alredy sampled ({window.start}->{window.end})"
                    )
                    break

        # This should NOT be triggered
        assert not foundSampled, "Couldnt find a non sampled window"

        self.sampled_window_count += 1
        self.sampled_windows.append(
            TimeWindow(
                start=local_starting_time, end=local_starting_time + window_length
            )
        )

        # Sample sequence as we normally would
        return super().sample(
            local_starting_time, window_skip, window_length, initial_precise
        )

    def clear_memory(self):
        self.sampled_windows = []


def binary_search(cpt_rdr: AbstractTimeSeriesReader, target_time: float) -> int:
    """
    Given a reader and a target time, will return the index of the first sample
    that is after the target time
    (i.e. excludes boundary)
    """
    # Initialize variables
    # self.logger.debug(f"The initial high is  {high}")
    low, high = binary_till_two(target_time, cpt_rdr)
    # Once we have two closest elements we check the closes
    # Argmax it
    if target_time <= float(cpt_rdr.getTimestamp(low)) and abs(
        target_time - float(cpt_rdr.getTimestamp(low))
    ) < abs(target_time - float(cpt_rdr.getTimestamp(high))):
        return low
    else:
        return high


def binary_search_upper(cpt_rdr: AbstractTimeSeriesReader, target_time: float) -> int:
    _, high = binary_till_two(target_time, cpt_rdr)
    return high


def binary_till_two(
    target_time: float, pckt_rdr: AbstractTimeSeriesReader
) -> Tuple[int, int]:
    """
    Will do binary search until only two elements remain
    """
    low: int = 0
    high: int = len(pckt_rdr) - 1
    mid: int = 0
    # Do binary search
    # while high > low:
    while (high - low) != 1:
        mid = ceil((high + low) / 2)  # CHECK: It *should* be ceil. Check nonetheless
        if target_time > pckt_rdr.getTimestamp(mid):
            low = mid
        else:
            high = mid
    return low, high
