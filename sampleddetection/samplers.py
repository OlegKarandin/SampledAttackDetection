import logging
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Generic, List, Sequence, Tuple, TypeVar

import numpy as np

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
        specific_samplefactory: SampleFactory,  # TOREM: the reader can take on this complexity this is redundant.
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
        self.logger.debug(
            f"Entering with starting_time {starting_time} (of type {type(starting_time)}), window_skip {window_skip} and window_length {window_length}"
        )

        _starting_time = starting_time
        _stopping_time = starting_time + window_length

        samples = []
        for s in range(self.sampling_budget):
            idx_firstsamp = binary_search(self.timeseries_rdr, _starting_time)
            idx_lastsamp = binary_search(self.timeseries_rdr, _stopping_time)

            self.logger.debug(
                f"Entering with starting_time {_starting_time} and ending at {_stopping_time}"
            )
            self.logger.debug(f"Will sample between {idx_firstsamp} -> {idx_lastsamp}")

            # This call might be IPC so be careful not to abuse it
            samples += self.timeseries_rdr[idx_firstsamp:idx_lastsamp]
            self.logger.debug(f"This contains {len(samples)}")
            _starting_time += window_skip
            _stopping_time = _starting_time + window_length

        return samples


class NoReplacementSampler(DynamicWindowSampler):
    def __init__(
        self,
        csvrdr: AbstractTimeSeriesReader,
        specific_sample_factory: SampleFactory,
        lowest_resolution: float = 1e-6,
    ):
        super().__init__(csvrdr, specific_sample_factory, lowest_resolution)
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
