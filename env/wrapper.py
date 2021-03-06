'''
Implemented by ghliu
https://github.com/ghliu/pytorch-ddpg/blob/master/normalized_env.py
'''
from abc import ABC

import gym


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper, ABC):
    """ Wrap action """

    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env=env)
        self.env = env
        self.action_high = 1.
        self.action_low = -1.

    def action(self, action):
        if not self.env.discrete_action_space:
            act_k = (self.action_high - self.action_low) / 2.
            act_b = (self.action_high + self.action_low) / 2.
            return act_k * action + act_b
        else:
            return action

    @property
    def agents(self):
        return self.env.agents

    @property
    def episode_length(self):
        return self.env.episode_length
