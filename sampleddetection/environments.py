import datetime
import random
from logging import DEBUG
from typing import Any, Sequence, Tuple, Union

import numpy as np

from sampleddetection.datastructures import Action, State
from sampleddetection.reward_signals import RewardCalculatorLike
from sampleddetection.samplers import FeatureFactory, TSSampler
from sampleddetection.utils import clamp, pretty_print, setup_logger, within


class SamplingEnvironment:
    """
    Responsibilities:
        Basic sampling of environments using window length and window frequency.
    State is defined as:
        A point in time together with current window size and current frequency
    """

    # Hyperparameters
    # CHECK: That we hav good ranges
    WINDOW_SKIP_RANGE = [1e-6, 150e-3]
    WINDOW_LENGTH_RANGE = [1e-6, 1e-2]
    AMOUNT_OF_SAMPLES_PER_ACTION = 1  # Where action means selection of frequency/window
    PREVIOUS_AMNT_SAMPLES = 4
    DAY_RIGHT_MARGIN = 1  # CHECK: Must be equal 1 at deployment

    def __init__(
        self,
        sampler: TSSampler,
        feature_factory: FeatureFactory[Any],
        reward_calculator: RewardCalculatorLike,
    ):
        self.sampler = sampler
        self.logger = setup_logger(self.__class__.__name__, DEBUG)
        # self.observable_features = observable_features
        self.feature_factory = feature_factory
        self.reward_calculator = reward_calculator

        # Internal representation of State. Will be returned to viewer as in `class State` language
        self.starting_time = float("-inf")
        self.cur_state = State(
            time_point=float("-inf"),
            window_skip=float("-inf"),
            window_length=float("-inf"),
            observations=np.ndarray([0]),
        )
        # CHECK: that this reset call is even necessary
        self.reset()

    def step(self, action: Action) -> Tuple[State, float, dict]:
        """
        Core functionality as well as data processing b4

        returns
        ~~~~~~~
            State: new state consequent to input `action`
            Reward: reward for taking `action` conditioned on stored state
        """
        # Ensure correct initialization.
        status = [
            self.cur_state.time_point > 0,
            self.cur_state.window_skip > 0,
            self.cur_state.window_length > 0,
        ]
        # self.logger.debug(
        #     f"Starting cur_time is {self.cur_state.time_point} of type {type(self.cur_state.time_point)}"
        # )
        assert all(status), (
            "Make sure you initialize enviornment properly\n Specifically:"
            f"\n\ttime_point={self.cur_state.time_point}"
            f"\n\twindow_skip={self.cur_state.window_skip}"
            f"\n\twindow_length={self.cur_state.window_length}"
        )

        # self.logger.debug(
        #     f"The action is of type {type(action)} and looks like {action} with element-type {type(action.winlen_delta)}"
        # )

        ### Preprocess data for new state
        window_skip = clamp(
            self.cur_state.window_skip + action.winskip_delta,
            self.WINDOW_SKIP_RANGE[0],
            self.WINDOW_SKIP_RANGE[1],
        )
        cur_time = self.cur_state.time_point + self.cur_state.window_skip
        # self.logger.debug(
        #     f"window_skip ({window_skip}) itself is of type {type(window_skip)}"
        # )
        # self.logger.debug(
        #     f"After window_skip addition  cur_time is {cur_time} of type {type(cur_time)}"
        # )
        self.cur_state.time_point = cur_time
        # Onbserve after the rest
        window_length = clamp(
            self.cur_state.window_length + action.winlen_delta,
            self.WINDOW_LENGTH_RANGE[0],
            self.WINDOW_LENGTH_RANGE[1],
        )

        ### ✨ Time to observe (take a step)
        # self.logger.debug(
        #     f"Sampling about to take place at current time {cur_time} (of type {type(cur_time)}) with window lenght {window_length}"
        # )
        # TODO: Ensure we can remove window_skipo later, its not being used already
        new_samples = self.sampler.sample(
            cur_time, window_skip, window_length, first_sample=False
        )
        # if len(new_samples) >= 0:
        #     self.logger.debug(f"Obtained {len(new_samples)} from our sampler")
        # else:
        #     self.logger.warn(f"Length of samples is 0")
        #
        arraylike_features, labels = self.feature_factory.make_feature_and_label(
            new_samples
        )
        self.logger.debug(
            f"We got {len(labels)} samples at cur_time {cur_time} with window_skip {window_skip} and window_length {window_length}"
        )

        # self.logger.debug("In preparation to go into State")
        # Update the state to new observations
        new_state = State(
            # Time point at which next step will start
            time_point=cur_time + window_length,
            window_skip=window_skip,
            window_length=window_length,
            observations=arraylike_features,
        )

        ### START: Debug section
        # Get statistics of iat between between first and last sampling points
        # Should give us an idea of where our algorithm resides to be.
        extra_obs = {}
        # extra_obs = self.sampler.window_statistics(cur_time, window_skip, window_length)

        # Show current staste
        _debug_ts = datetime.datetime.fromtimestamp(cur_time)
        _debug_ts_str = _debug_ts.strftime("%H:%M:%S")
        self.logger.debug(
            f"Our current window_skip {window_skip} and window length {window_length} for timepoint {_debug_ts_str}"
        )

        ### END: Debug section
        return_reward = self.reward_calculator.calculate(
            features=arraylike_features, ground_truths=labels
        )

        ### Update new state
        self.cur_state = new_state

        return self.cur_state, return_reward, extra_obs

    def reset(
        self,
        starting_time: Union[float, None] = None,
        winskip: Union[float, None] = None,
        winlen: Union[float, None] = None,
    ) -> State:
        # Find different simulatenous positions to draw from
        # Staring Frequencies
        # Create New Flow Session

        self._initialize_triad(starting_time, winskip, winlen)

        samples: Sequence = self.sampler.sample(
            self.cur_state.time_point,
            self.cur_state.window_skip,
            self.cur_state.window_length,
            first_sample=True,
        )

        # self.logger.debug(f"Sampled for SamplingEnvironment reset")

        arraylike_features, label = self.feature_factory.make_feature_and_label(samples)

        return_state = State(
            time_point=self.cur_state.time_point,
            window_skip=self.cur_state.window_skip,
            window_length=self.cur_state.window_length,
            observations=arraylike_features,
            # observable_features=self.observable_features,
        )
        return return_state

    def _initialize_triad(
        self,
        starting_time: Union[None, float] = None,
        winskip: Union[None, float] = None,
        winlen: Union[None, float] = None,
    ):
        """
        Will initialize the starting_time, winskin, winlen according to whether or not the provided paremeters are empty

        Variables (3)
        ---------
        - starting_time: Time at which this environment is to start taking steps
        - winskip: Once placed at `starting_time` how long will agent not observe for until the next observation
        - winlen: Length of new obserbatio

        Returns (0)
        ---------
        Returns nothing but sets a new `self.cur_state`
        """

        # self.logger.debug("Initializing triad")
        min_time, max_time = (
            self.sampler.init_time,
            self.sampler.fin_time,
        )

        # self.logger.debug("Restarting the environment")
        assert min_time != max_time, "Cap Reader not initialized Properly"

        # Starting Time
        if starting_time == None:
            # self.logger.debug(f"Unset, setting")
            self.cur_state.time_point = random.uniform(
                min_time,
                min_time + (max_time - min_time) * self.DAY_RIGHT_MARGIN,
            )
            # self.logger.debug(f"Unset, ensuring {type(self.cur_state.time_point)}")
        else:
            assert within(
                starting_time, min_time, max_time * self.DAY_RIGHT_MARGIN
            ), f"Stating time {starting_time} out of range [{min_time},{max_time}]"
            # self.logger.debug(f"Preset, setting")
            self.cur_state.time_point = starting_time

        # self.logger.debug(
        #     f"Initialized starting_time to {self.cur_state.time_point} with type {type(self.cur_state.time_point)}"
        # )
        # Winskip
        if winskip == None:
            self.cur_state.window_skip = random.uniform(
                self.WINDOW_SKIP_RANGE[0], self.WINDOW_SKIP_RANGE[1]
            )
        else:
            assert within(
                winskip, self.WINDOW_SKIP_RANGE[0], self.WINDOW_SKIP_RANGE[1]
            ), f"Winskip {winskip} out of range"
            self.cur_state.window_skip = winskip

        # Window Length initialization
        if winlen == None:
            self.cur_state.window_length = random.uniform(
                self.WINDOW_LENGTH_RANGE[0], self.WINDOW_LENGTH_RANGE[1]
            )
        else:
            assert within(
                winlen, self.WINDOW_LENGTH_RANGE[0], self.WINDOW_LENGTH_RANGE[1]
            ), f"Winlen {winlen} out of range"
            self.cur_state.window_length = winlen
