import math
import random
from logging import DEBUG
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from sampleddetection.common_lingo import Action, State
from sampleddetection.datastructures.flowsession import SampledFlowSession
from sampleddetection.environment.agents import AgentLike, BaselineAgent, RLAgent
from sampleddetection.samplers.window_sampler import (
    DynamicWindowSampler,
    UniformWindowSampler,
)

from ..utils import clamp, setup_logger, within


class Environment:
    """
    State is defined as:
        A point in time together with current window size and current frequency
    """

    # Hyperparameters
    # CHECK: That we hav good ranges
    WINDOW_SKIP_RANGE = [1e-7, 1e2]
    WINDOW_LENGTH_RANGE = [1e-5, 1e2]
    AMOUNT_OF_SAMPLES_PER_ACTION = 1  # Where action means selection of frequency/window
    PREVIOUS_AMNT_SAMPLES = 12
    FLOW_MEMORY = 12  # Per-flow packet budget
    MIN_FLOW_MEMORY = 9999999999  # Maximum amount of flows to store.
    DAY_RIGHT_MARGIN = 0.8  # CHECK: Must be equal 1 at deployment

    def __init__(
        self,
        sampler: DynamicWindowSampler,
    ):
        self.sampler = sampler
        self.logger = setup_logger(self.__class__.__name__, DEBUG)

        # Internal representation of State. Will be returned to viewer as in `class State` language
        self.starting_time = float("-inf")
        self.cur_winlen = float("-inf")
        self.cur_winskip = float("-inf")

    def step(self, cur_state: State, action: Action) -> Tuple[State, float]:
        """
        returns
        ~~~~~~~
            State:  State
            Reward: float
        """
        status = [self.starting_time > 0, self.cur_winskip > 0, self.cur_winskip > 0]
        assert all(status), "Make sure you initialize enviornment properly"

        # Get current positions
        self.cur_winskip = clamp(
            self.cur_winskip + action.winskip_delta,
            self.WINDOW_SKIP_RANGE[0],
            self.WINDOW_SKIP_RANGE[1],
        )
        self.cur_winlen = clamp(
            self.cur_winlen + action.winlength_delta,
            self.WINDOW_LENGTH_RANGE[0],
            self.WINDOW_SKIP_RANGE[1],
        )

        return self._step(self.starting_time, self.cur_winskip, self.cur_winlen)

    def _step(self, start_time, winskip, winlen) -> Tuple[State, float]:
        """
        returns
        ~~~~~~~
            State:  State
            Reward: float
        """
        # Do Sampling
        flow_sesh = self.sampler.sample(start_time, winskip, winlen)

        return_state = State(
            time_point=start_time,
            window_skip=winskip,
            window_length=winlen,
            flow_sesh=flow_sesh,
        )

        # TODO: calculatae the reward
        return_reward = 0

        return return_state, return_reward

    def reset(
        self,
        starting_time: Union[None, float] = None,
        winskip: Union[None, float] = None,
        winlen: Union[None, float] = None,
    ) -> State:
        # Find different simulatenous positions to draw from
        # Staring Frequencies
        # Create New Flow Session

        self._initialize_triad(starting_time, winskip, winlen)

        flow_sesh = self.sampler.sample(
            self.starting_time,
            self.cur_winskip,
            self.cur_winlen,
        )

        return State(
            time_point=self.starting_time,
            window_skip=self.cur_winskip,
            window_length=self.cur_winlen,
            flow_sesh=flow_sesh,
        )

    def _initialize_triad(
        self,
        starting_time: Union[None, float] = None,
        winskip: Union[None, float] = None,
        winlen: Union[None, float] = None,
    ):
        min_time, max_time = (
            self.sampler.csvrdr.first_sniff_time,
            self.sampler.csvrdr.last_sniff_time,
        )

        self.logger.debug("Restarting the environment")
        assert min_time != max_time, "Cap Reader not initialized Properly"

        # Starting Time
        if starting_time == None:
            self.starting_time = random.uniform(
                min_time,
                min_time + (max_time - min_time) * self.DAY_RIGHT_MARGIN,
            )
        else:
            assert within(
                starting_time, min_time, max_time * self.DAY_RIGHT_MARGIN
            ), f"Stating time {starting_time} out of range [{min_time},{max_time}]"
            self.starting_time = starting_time

        # Winskip
        if winskip == None:
            self.cur_winskip = random.uniform(
                self.WINDOW_SKIP_RANGE[0], self.WINDOW_SKIP_RANGE[1]
            )
        else:
            assert within(
                winskip, self.WINDOW_SKIP_RANGE[0], self.WINDOW_SKIP_RANGE[1]
            ), f"Winskip {winskip} out of range"
            self.cur_winskip = winskip

        if winlen == None:
            self.cur_winlen = random.uniform(
                self.WINDOW_LENGTH_RANGE[0], self.WINDOW_LENGTH_RANGE[1]
            )
        else:
            assert within(
                winlen, self.WINDOW_LENGTH_RANGE[0], self.WINDOW_LENGTH_RANGE[1]
            ), f"Winlen {winlen} out of range"
            self.cur_winlen = winlen
