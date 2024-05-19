from pathlib import Path
from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType

from networking.netfactories import NetworkFeatureFactory, NetworkSampleFactory
from sampleddetection.datastructures import State
from sampleddetection.environments import SamplingEnvironment
from sampleddetection.readers import CSVReader
from sampleddetection.samplers import (
    FeatureFactory,
    NoReplacementSampler,
    SampleFactory,
)


class GymSamplingEnv(gym.Env):
    """
    Responibilities:
        Wrap around non-gym environment
    """

    metadata = {}

    def __init__(
        self,
        csv_path_str: str,
        num_obs_elements: int,
        num_possible_actions: int,
        action_idx_to_direction: Dict[int, int],  # TODO: maybe change to simply scaling
        sample_factory: SampleFactory,
        feature_factory: FeatureFactory,
    ):

        # Get Dependency Injection elements.
        csv_path = Path(csv_path_str)
        csv_reader = CSVReader(csv_path)
        # TODO: we have to check this NoReplacementSampler is not too slow
        meta_sampler = NoReplacementSampler(csv_reader, sample_factory)

        self.env = SamplingEnvironment(meta_sampler, feature_factory=feature_factory)

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
        self.action_space = gym.spaces.Discrete(num_possible_actions)
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # CHECK: if these are even necessary
        self.render_mode = None
        self.window = None
        self.clock = None

    def step(self, action: ActType):
        """
        Must return a 5-tuple in accordance to gym.Env
        """
        # CHECK: since action_idx is of type ActType
        # maybe we have to check its provenance
        actual_action = int(self.action_idx_to_direction[action])

        # Then we get the action environment to act on it
        state, reward = self.env.step(actual_action)
        # Tuple to return is (observation, reward, terminated, truncaed, info)

        # We must convv
        return self.env(action)

    def _get_obs(self, state: State):
        # State will return a list so we just have to return that as well
        return state.get_data_list(self.relevant_datapoints)

    def reset(self, seed=None, options=None):
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
