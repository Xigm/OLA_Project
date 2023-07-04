# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 04:28:55 2023

@author: saade
"""

import numpy as np
import matplotlib.pyplot as plt 
from bidding_environnement_stp2 import *
from GPTS_Learner import *

n_arms = 100
min_bid = 0
max_bid = 1

bids = np.linspace(min_bid,max_bid, n_arms)
sigma = 10

T = 365
n_experiments = 500

gpts_rewards_per_experiment = []


#just an arbitrary value since we need to get it from a given curve 
M = 50  
cr = 0.6


for e in range(0, n_experiments):
    env = BiddingEnvironment(bids=bids, sigma=sigma, margin=M, Conv_rate=cr)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=bids)
    
    for t in range(0,T):
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm, reward)
    
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    
    
    
opt = np.max(env.means)
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")

plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0)),'r')
        

plt.show()       


