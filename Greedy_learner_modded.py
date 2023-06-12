# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:01:07 2023
@author: Xigm

"""

from learner_super import learner
import numpy as np


class Greedy_learner_modded(learner):
    def __init__(self, n_arms, max_prices):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)
        self.max_prices = max_prices
        
    def pull_arm(self):
        if (self.t < self.n_arms):
            return self.t
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
        
    def update(self,pulled_arm,reward):
        self.t = self.t +1
        self.update_observations(pulled_arm, reward)
        reward_com = reward/self.max_prices
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm]*(self.t-1)+reward_com)/self.t