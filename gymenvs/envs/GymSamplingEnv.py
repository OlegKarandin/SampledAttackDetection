from pathlib import Path
from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import numpy as np
import ray
from gymnasium.core import ActType
from ray import ObjectRef

from networking.netfactories import NetworkFeatureFactory, NetworkSampleFactory
from sampleddetection.datastructures import Action, State
from sampleddetection.environments import SamplingEnvironment
from sampleddetection.readers import AbstractTimeSeriesReader, CSVReader
from sampleddetection.reward_signals import RewardCalculatorLike
from sampleddetection.samplers import (
    FeatureFactory,
    NoReplacementSampler,
    SampleFactory,
)
from sampleddetection.utils import setup_logger


class GymSamplingEnv(gym.Env):
    """
    Responibilities:
        Wrap around non-gym environment
    """

    metadata = {}

    def __init__(
        self,
        num_obs_elements: int,
        num_possible_actions: int,
        # data_reader: AbstractTimeSeriesReader,
        data_reader_ref: ObjectRef,
        action_idx_to_direction: Dict[int, int],  # TODO: maybe change to simply scaling
        sample_factory: SampleFactory,
        feature_factory: FeatureFactory,
        reward_calculator: RewardCalculatorLike,
    ):

        # Get Dependency Injection elements.
        self.logger = setup_logger(__class__.__name__)
        # TODO: we have to check this NoReplacementSampler is not too slow
        self.data_reader = ray.get(data_reader_ref)
        meta_sampler = NoReplacementSampler(self.data_reader, sample_factory)

        self.env = SamplingEnvironment(
            meta_sampler,
            reward_calculator=reward_calculator,
            feature_factory=feature_factory,
        )

        # TOREM: Retrieve feature_factories observation dictionary
        # obs_el_str = feature_factory.make_feature_and_label()
        # self.logger.debug(f"Reporting on the features that we are seeing: {obs_el_str}")

        self.action_idx_to_direction = action_idx_to_direction

        n = num_obs_elements
        low_value = -1.0  # Replace with the minimum value for each element
        high_value = 1.0  # Replace with the maximum value for each element
        # TODO: Actually define the limtis here
        self.observation_space = gym.spaces.Box(
            low=np.full((n,), low_value, dtype=np.float32),
            high=np.full((n,), high_value, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([num_possible_actions, num_possible_actions]),
            dtype=np.float32,
        )
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # CHECK: if these are even necessary
        self.render_mode = None
        self.window = None
        self.clock = None

    # def step(self, action: ActType):
    def step(self, action: np.ndarray):
        """
        Interface into applications that use Gym environments.

        Variables (1)
        ---------
            - action: a 2-element vector containing winskip_delta and winskip_delta (in that order)

        Returns (5)
        ---------
            - observation (dict)
            - reward (float)
            - terminated (bool): Whether it has terminated or not
            - trucated (bool): Not sure
            - info (dict): unsure
        """

        # CHECK: since action_idx is of type ActType
        # maybe we have to check its provenance
        self.logger.debug(
            f"Gym Envirnonment is receiving action type ({type(action)}) that looks like: {action}"
        )
        # actual_action = int(self.action_idx_to_direction[action])

        # Create Action in same language
        action_message = Action(winskip_delta=action[0], winlen_delta=action[1])

        # Then we get the action environment to act on it
        state, reward = self.env.step(action_message)
        # Tuple to return is (observation, reward, terminated, truncaed, info)

        terminated = False
        truncated = False
        info = {}

        observation = state.observations.mean(axis=0)

        self.logger.debug(f"Observation is looking like {observation}")
        # We must convv
        return observation, reward, terminated, truncated, info

    # def _get_obs(self, state: State):
    #     # State will return a list so we just have to return that as well
    #     return state.get_data_list(self.relevant_datapoints)

    def reset(
        self,
        *args,
        seed: Union[int, None] = None,
        options: Union[dict[str, Any], None] = None,
    ):
        super().reset(seed=seed)  # type:ignore
        # Return observations as are expected

        # Form observation and form information
        state: State = self.env.reset()

        obs_shape = state.observations.shape
        assert len(obs_shape) == 2, "Incorrect observation shape."

        # TODO: Think about whether or not we want to leave it like this.
        aggregated_observation = state.observations.mean(axis=0)

        info = {}  # CHECK: If we can actually use it for something.

        return aggregated_observation, info

    # Perhaps add a `render` method for visualization.
