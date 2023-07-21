# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:39:24 2023
@author: Xigm

"""


from learner_super import learner
import numpy as np

class TS_Learner_modded(learner):
    def __init__(self,n_arms, max_prices):
        super().__init__(n_arms)
        self.beta_params = np.ones((n_arms,2))
        self.max_prices = max_prices
        
    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_params[:,0],self.beta_params[:,1]))
        return idx
    
    def update(self,pulled_arm,reward):
        self.t = self.t+1
        self.update_observations(pulled_arm, reward)
        reward_bernoulli = reward/self.max_prices
        self.beta_params[pulled_arm,0] = self.beta_params[pulled_arm,0] + reward_bernoulli
        self.beta_params[pulled_arm,1] = self.beta_params[pulled_arm,1] + 1 - reward_bernoulli