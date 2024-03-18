from logging import DEBUG
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

from sampleddetection.common_lingo import Action, State
from sampleddetection.datastructures.flowsession import SampledFlowSession
from sampleddetection.samplers.window_sampler import (
    DynamicWindowSampler,
    UniformWindowSampler,
)

from ..utils import setup_logger


class Environment:
    """
    State is defined as:
        A point in time together with current window size and current frequency
    """

    # Hyperparameters
    # CHECK: That we hav good ranges
    WINDOW_SKIP_RANGE = [1e-3, 1e2]
    WINDOW_LENGTH_RANGE = [1e-6, 1e-5]
    # TODO: Implement below
    AMOUNT_OF_SAMPLES_PER_ACTION = 1  # Where action means selection of frequency/window
    PREVIOUS_AMNT_SAMPLES = 12
    FLOW_MEMORY = 12  # Per-flow packet budget
    MIN_FLOW_MEMORY = 9999999999  # Maximum amount of flows to store.

    def __init__(
        self,
        sampler: DynamicWindowSampler,
        simultaneous_enviroments: int,
    ):
        self.sampler = sampler
        self.M = simultaneous_enviroments
        self.logger = setup_logger(self.__class__.__name__, DEBUG)

    def step(self, cur_state: State, action: Action) -> torch.Tensor:
        # Get current positions
        new_freq = cur_state.cur_frequency + action.frequency_delta
        new_win = cur_state.window_length + action.window_length_delta
        new_freq = max(
            self.WINDOW_SKIP_RANGE[0], min(self.WINDOW_SKIP_RANGE[1], new_freq)
        )
        new_win = max(
            self.WINDOW_LENGTH_RANGE[0], min(self.WINDOW_LENGTH_RANGE[1], new_win)
        )

        # Calculate Rewards
        new_state = self.sampler  # TODO: finish this

        return torch.Tensor([])  # TOREM: remove this in favor of something meaningful

    def reset(self, starting_times: Union[None, Tensor] = None) -> List[State]:
        # Find different simulatenous positions to draw from
        min_time, max_time = (
            self.sampler.csvrdr.first_sniff_time,
            self.sampler.csvrdr.last_sniff_time,
        )

        self.logger.debug("Restarting the environment")
        assert min_time != max_time, "Cap Reader not initialized Properly"

        # Select M distinct staring positions
        # TODO: Ensure we are not selecting to far into the day where no samples are possible.
        # ( We could also just leave it as noise for now >:])
        if starting_times == None:
            starting_times = torch.rand(self.M) * (max_time - min_time) + min_time
        else:
            assert (
                len(starting_times) == self.M
            ), "Starting_times not equal to amount of environments"
        self.logger.debug(f"Staring times are {starting_times}")

        # Staring Frequencies
        starting_winskips = (
            torch.rand(self.M) * (self.WINDOW_SKIP_RANGE[1] - self.WINDOW_SKIP_RANGE[0])
            + self.WINDOW_SKIP_RANGE[0]
        )
        starting_winlens = (
            torch.rand(self.M)
            * (self.WINDOW_LENGTH_RANGE[1] - self.WINDOW_LENGTH_RANGE[0])
            + self.WINDOW_LENGTH_RANGE[0]
        )

        # Create New Flow Session
        states: List[State] = []
        for i in range(self.M):
            flow_sesh = self.sampler.sample(
                starting_times[i].item(),
                starting_winskips[i].item(),
                starting_winlens[i].item(),
            )
            states.append(
                State(
                    time_point=starting_times[i].item(),
                    cur_frequency=starting_winskips[i].item(),
                    window_length=starting_winlens[i].item(),
                    flow_sesh=flow_sesh,
                )
            )

        return states
