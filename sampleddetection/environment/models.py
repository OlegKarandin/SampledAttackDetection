import math
import random
from logging import DEBUG
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# from .agents import AgentLike, BaselineAgent, RLAgent
from sampleddetection.samplers.window_sampler import (DynamicWindowSampler,
                                                      UniformWindowSampler)

from ..datastructures.flowsession import SampledFlowSession
from ..utils import clamp, setup_logger, within
from .datastructures import Action, State


class Environment:
    """
    State is defined as:
        A point in time together with current window size and current frequency
    """

    # Hyperparameters
    # CHECK: That we hav good ranges
    WINDOW_SKIP_RANGE = [1e-7, 1e2]
    WINDOW_LENGTH_RANGE = [1e-6, 1e2]
    AMOUNT_OF_SAMPLES_PER_ACTION = 1  # Where action means selection of frequency/window
    PREVIOUS_AMNT_SAMPLES = 12
    FLOW_MEMORY = 12  # Per-flow packet budget
    MIN_FLOW_MEMORY = 9999999999  # Maximum amount of flows to store.
    DAY_RIGHT_MARGIN = 1  # CHECK: Must be equal 1 at deployment

    def __init__(
        self,
        sampler: Union[DynamicWindowSampler, None],
    ):
        self.sampler = sampler if sampler != None else: 
        self.logger = setup_logger(self.__class__.__name__, DEBUG)

        # Internal representation of State. Will be returned to viewer as in `class State` language
        self.starting_time = float("-inf")
        self.cur_winlen = float("-inf")
        self.cur_winskip = float("-inf")

    def step(self, action: Action) -> Tuple[State, float]:
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
            cur_window_skip=winskip,
            cur_window_length=winlen,
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
            cur_window_skip=self.cur_winskip,
            cur_window_length=self.cur_winlen,
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


class Agent:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        observable_datapoints: List[str],
    ):
        # self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1)
        self.onbservable_datapoints = observable_datapoints
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.nn = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def query(self, state: State) -> Action:
        # TODO: Try a summarry og history of states if we find it necessary
        # Lets disect the state
        state_tensor = state.as_tensor(self.onbservable_datapoints)

        nn_result = self.nn(state_tensor)
        action = Action(nn_result)

        return action


class ExperienceBuffer:
    """
    Meant to store all experiences we collect with on-policy approach
    For now it will store a fixed amount of experiences
    Will use the Generalized Advantage Estimator (GAE) for estimation of advantage function
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        path_start_idx: int,
        buffer_size: int,
        lam: float,
    ):
        assert (0 < lam) and (lam < 1), "Lambda must be in (0,1)"
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer_size = buffer_size
        self.path_start_idx = path_start_idx

        # Create the buffers
        self.obs_buff = torch.zeros((buffer_size, obs_dim))
        self.act_buff = torch.zeros(buffer_size, act_dim)

        self.idx = 0

    def store(self, new_obs: torch.Tensor, new_acts: torch.Tensor):
        self.obs_buff[self.idx] = new_obs
        self.act_buff[self.idx] = new_acts

        self.idx += 1

    def compute_advantage_estimate(self):
        assert self.idx == self.buffer_size, "Buffer not full."
        # Compute Rewards to Go
        path_slice = slice(self.path_start_idx, self.idx)

        # Compute Advantage Estimates


class Critic(nn.Module):
    def __init__(self, sizes: Tuple[int, ...]):
        super().__init__()
        num_layers = len(sizes) - 1
        self.net_list: List[nn.Module] = []
        for i in range(num_layers):
            self.net_list.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (num_layers):
                # Normal Relu Activation
                self.net_list.append(nn.ReLU())

        self.net = nn.Sequential(*self.net_list)

    def forward(self, obs):
        # CHECK: That we have the right shape
        return self.net(obs)


class GaussianActor(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        log_std = -0.5 * torch.ones(action_dim, dtype=torch.float32)

    def forward(self, obs: torch.Tensor):
        pass  # TODO:


class CategoricalActor(nn.Module):
    # def __init__(self, obs_dim: int, action_dim: int):
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...], action_dim: int):
        super().__init__()
        # Will jus be a softmax that we get out of this.
        # This will be passed to be sampled as catgeorical
        assert len(hidden_sizes) >= 1, "Not enough hidden dimensions"
        self.obs_dim = obs_dim

        net_list = [nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU()]

        meta_hidden_sizes = hidden_sizes + (action_dim,)

        for h in range(len(meta_hidden_sizes) - 1):
            net_list.append(nn.Linear(hidden_sizes[h], hidden_sizes[h + 1]))
            if h != (len(meta_hidden_sizes) - 1):
                net_list.append(nn.ReLU())

        self.net = nn.Sequential(*net_list)

    def get_dist(self, x: torch.Tensor) -> Categorical:
        # Assert dimensions of x
        assert (
            x.shape[1] == self.obs_dim
        ), "Input shape not matching for CategoricalActor"

        logits = self.net(x)
        return Categorical(logits=logits)


class ActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.critic = Critic((obs_dim,) + hidden_sizes)
        self.actor = CategoricalActor(obs_dim, hidden_sizes, action_dim)

    def step(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Will return action, observastion value, and actions logprob

        Arguments
        ---------

        obs: torch.Tensor -> Obervartions

        Return
        ------
        Action, Value, LogProbs
        """
        # Optimization comes later
        with torch.no_grad():
            # Get the action
            action_dist = self.actor.get_dist(obs)
            action_sample = action_dist.sample()
            action_logprobs = action_dist.log_prob(action_sample)

            # Get the value of state
            obs_value = self.critic(obs)

        return action_sample, obs_value, action_logprobs
