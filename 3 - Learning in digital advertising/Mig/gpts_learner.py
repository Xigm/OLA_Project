# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 19:47:09 2023
@author: Xigm

"""

from learner_super import learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C



class gpts_learner(learner):
    def __init__(self, n_arms, arms, theta, alpha):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigma = np.ones(self.n_arms)*10
        self.pulled_arms = []
        kernel = C(theta, (1e-3,1e3)) * RBF(1,(1e-3,1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha, normalize_y=True, n_restarts_optimizer=5)
        
    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])
        
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)
        
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        
    def pull_arm(self):
        sampled_values = np.random.normal(self.means,self.sigma)
        return np.argmax(sampled_values)
    
    def score(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        return self.gp.score(x,y)
        