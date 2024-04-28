#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: action repeat wrappers for data and model type environment
#  Update: 2022-11-15, Wenxuan Wang: create action repeat wrapper


from __future__ import annotations

from typing import TypeVar, Tuple, Union
from gym.spaces import Discrete, MultiDiscrete

import gym
import torch
import numpy as np
from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class DiscreteActionData(gym.Wrapper):
    """Action repeat wrapper fot data type environments, repeat 'repeat_num' times
        action in one step and return last step observation.

    :param env: data type environment.
    :param int repeat_num: repeat n times action in one step.
    :param bool sum_reward: sum the rewards during repeating steps, if set to False,
        only use reward in last step.
    """

    def __init__(self, env, action_num_per_dim: int = 5, multi_discrete = True):
        super(DiscreteActionData, self).__init__(env)
        action_dim = (
            env.action_space.shape[0]
            if len(env.action_space.shape) == 1
            else env.action_space.shape
        )
        self.steer_low = self.env.action_space.low[0]
        self.steer_high = self.env.action_space.high[0]
        self.action_space = Discrete(action_num_per_dim)

    def step(self, action_dis: ActType) -> Tuple[ObsType, float, bool, dict]:
        action_conti = self.action_discre2conti(action_dis)
        obs, r, d, info = self.env.step(action_conti)
        return obs, r, d, info
    
    def action_discre2conti(self, action_dis):
        # print("self.env.action_space.nvec", self.env.action_space.nvec)
        # print("action_dis", action_dis)
        ratio = action_dis / (self.action_space.n - 1)
        steer = self.steer_low + ratio * (self.steer_high - self.steer_low)
        action_conti = np.array([steer, 0.0])
        # print("action_conti", action_conti)
        return action_conti
