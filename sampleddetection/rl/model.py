from typing import List, NamedTuple

import torch
import torch.nn as nn

from sampleddetection.samplers.window_sampler import UniformWindowSampler


class State(NamedTuple):
    time_point : float
    window_length : float
    cur_frequency : float

class Action(NamedTuple):
    window_length_delta : float
    frequency_delta : float

def Environment():
    """
    State is defined as:
        A point in time together with current window size and current frequency
    """

    def __init__(
        self,
        sampler: UniformWindowSampler,
        desired_features : List[string]):

        self.sampler = sampler

    def step(self,cur_state: State, action: Action) :
        # 
        

    
