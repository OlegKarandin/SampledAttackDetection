from pathlib import Path
from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType

from sampleddetection.datastructures import Action, State
from sampleddetection.environments import SamplingEnvironment
from sampleddetection.readers.readers import CSVReader
from sampleddetection.samplers import NoReplacementSampler


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
        observable_features: Sequence,
    ):

        # Get Sampler
        csv_path = Path(csv_path_str)
        # TODO: we have to check this NoReplacementSampler is not too slow
        # Create the Reader
        csv_reader = CSVReader(csv_path)
        meta_sampler = NoReplacementSampler(csv_reader)

        self.env = SamplingEnvironment(meta_sampler, observable_features)

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
        return self.env.reset()

    # Perhaps add a `render` method for visualization.
