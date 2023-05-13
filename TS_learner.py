# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:50:01 2023
@author: Xigm

"""

from learner_super import learner
import numpy as np

class TS_Learner(learner):
    def __init__(self,n_arms):
        super().__init__(n_arms)
        self.beta_params = np.ones((n_arms,2))
        
    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_params[:,0],self.beta_params[:,1]))
        return idx
    
    def update(self,pulled_arm,reward):
        self.t = self.t+1
        self.update_observations(pulled_arm, reward)
        self.beta_params[pulled_arm,0] = self.beta_params[pulled_arm,0] + reward
        self.beta_params[pulled_arm,1] = self.beta_params[pulled_arm,1] + 1 - reward